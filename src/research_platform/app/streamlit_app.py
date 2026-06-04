"""Streamlit UI for quasi-experiment research platform."""

from io import BytesIO

import pandas as pd
import streamlit as st

from research_platform.core.method_guidance import METHOD_GUIDES, MethodGuide
from research_platform.core.insights import format_p_value
from research_platform.core.notebook_export import build_reproduction_notebook
from research_platform.core.profiling import profile_dataset
from research_platform.core.recommendation import recommend_method
from research_platform.core.reporting import result_to_markdown
from research_platform.core.schemas import (
    AnalysisMethod,
    AnomalyDetectionParams,
    CausalImpactParams,
    DidParams,
    EventStudyParams,
    ForecastBaselineParams,
    InterruptedTimeSeriesParams,
)
from research_platform.data_io.csv_loader import load_csv
from research_platform.methods.anomaly_detection import run_anomaly_detection
from research_platform.methods.causal_impact import run_causal_impact
from research_platform.methods.did import run_did
from research_platform.methods.event_study import run_event_study
from research_platform.methods.forecast_baseline import run_forecast_baseline
from research_platform.methods.interrupted_time_series import run_interrupted_time_series
from research_platform.utils.errors import ResearchPlatformError
from research_platform.utils.logging import setup_logging

setup_logging()

METHOD_LABELS = {
    AnalysisMethod.DID: "Diff-in-Diff",
    AnalysisMethod.EVENT_STUDY: "Event Study",
    AnalysisMethod.CAUSAL_IMPACT: "Causal Impact",
    AnalysisMethod.ITS: "Interrupted Time Series",
    AnalysisMethod.FORECAST_BASELINE: "Forecast + baseline",
    AnalysisMethod.ANOMALY_DETECTION: "Anomaly Detection",
}


def _dedupe_methods(methods: list[AnalysisMethod]) -> list[AnalysisMethod]:
    """Keep method order while removing duplicates."""
    seen: set[AnalysisMethod] = set()
    unique: list[AnalysisMethod] = []
    for method in methods:
        if method not in seen:
            unique.append(method)
            seen.add(method)
    return unique


def _init_state() -> None:
    defaults = {"df": None, "profile": None, "result": None, "step": 1, "scroll_to_top": False}
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _scroll_to_top_if_needed() -> None:
    """Scroll browser viewport to page top after wizard navigation."""
    if not st.session_state.get("scroll_to_top"):
        return
    st.session_state["scroll_to_top"] = False
    # st.html is not iframed, so the script runs in the app document directly
    # (replaces the deprecated st.components.v1.html).
    st.html(
        """
        <script>
          const doc = window.parent.document;
          const scrollTarget = doc.querySelector('[data-testid="stAppViewContainer"]');
          if (scrollTarget) {
            scrollTarget.scrollTo({ top: 0, behavior: "auto" });
          }
          window.parent.scrollTo({ top: 0, behavior: "auto" });
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def _go_to_step(step: int) -> None:
    """Move wizard to a specific step."""
    st.session_state["step"] = step
    st.session_state["scroll_to_top"] = True
    st.rerun()


def _back_button(label: str, step: int) -> None:
    """Render a back button."""
    if st.button(label):
        _go_to_step(step)


def _format_risk(risk_level: str) -> str:
    """Convert risk level to a Russian UI label with icon."""
    mapping = {
        "low": "🟢 Низкий",
        "medium": "🟡 Средний",
        "high": "🔴 Высокий",
    }
    return mapping.get(risk_level.lower(), risk_level)


def _format_decision(decision: str) -> str:
    """Convert decision hint to a Russian UI label with icon."""
    mapping = {
        "scale": "⬆️ Масштабировать",
        "hold_or_iterate": "↔️ Не масштабировать без доработки",
    }
    return mapping.get(decision, decision)


def _render_diagnostic(name: str, passed: bool, message: str) -> None:
    """Render diagnostic with colored Streamlit status."""
    text = f"**{name}**: {message}"
    if passed:
        st.success(f"✅ {text}")
    else:
        st.warning(f"⚠️ {text}")


def _render_method_guide(method: AnalysisMethod, guide: MethodGuide) -> None:
    """Render a method description card."""
    st.markdown(f"### {guide.title}")
    st.write(guide.short_description)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Когда использовать**")
        for item in guide.use_cases:
            st.write(f"- {item}")
        st.markdown("**Что нужно в данных**")
        for item in guide.required_data:
            st.write(f"- {item}")
    with c2:
        st.markdown("**Пример**")
        st.info(guide.example)
        st.markdown("**Минусы / ограничения**")
        for item in guide.limitations:
            st.write(f"- {item}")

    with st.expander(f"Что вернёт {METHOD_LABELS[method]}"):
        for item in guide.outputs:
            st.write(f"- {item}")
    if guide.differs_from:
        with st.expander("Чем отличается от похожих подходов", expanded=True):
            for item in guide.differs_from:
                st.write(f"- {item}")


def _render_selected_use_cases(method: AnalysisMethod) -> None:
    """Render concrete examples for the currently selected method."""
    guide = METHOD_GUIDES[method]
    st.markdown(f"**Если выбрать {guide.title}, типовые сценарии такие:**")
    for item in guide.use_case_examples:
        st.write(f"• {item}")


def _method_group(rec_method: AnalysisMethod, alternatives: list[AnalysisMethod]) -> list[AnalysisMethod]:
    """Build the visible comparison group: recommended method first, then alternatives."""
    return _dedupe_methods([rec_method, *alternatives])


def _render_method_comparison(
    *,
    group_methods: list[AnalysisMethod],
    recommended_method: AnalysisMethod,
    recommendation_reason: str,
) -> None:
    """Render all recommended/alternative methods in one compact comparison block."""
    st.subheader("Группа подходящих методов")
    st.info(
        "⭐ В приоритете сейчас: "
        f"**{METHOD_LABELS[recommended_method]}**. Почему: {recommendation_reason}"
    )

    comparison_rows = []
    for method in group_methods:
        guide = METHOD_GUIDES[method]
        comparison_rows.append(
            {
                "Метод": f"⭐ {guide.title}" if method == recommended_method else guide.title,
                "Когда использовать": guide.use_cases[0] if guide.use_cases else guide.short_description,
                "Что даёт": "; ".join(guide.outputs[:2]),
                "Главное ограничение": guide.limitations[0] if guide.limitations else "Требует проверки предпосылок.",
            }
        )
    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

    st.markdown("**Подробности по каждому методу группы**")
    columns = st.columns(2)
    for idx, method in enumerate(group_methods):
        guide = METHOD_GUIDES[method]
        with columns[idx % 2]:
            badge = "⭐ Рекомендованный" if method == recommended_method else "Альтернатива"
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="font-size:0.9rem; line-height:1.25">
                      <div style="font-weight:700; font-size:1rem; margin-bottom:0.1rem">{guide.title}</div>
                      <div style="opacity:0.72; margin-bottom:0.25rem">{badge}</div>
                      <div style="margin-bottom:0.45rem">{guide.short_description}</div>
                      <div><b>Пример:</b> {guide.example}</div>
                      <div style="margin-top:0.35rem"><b>Когда использовать:</b> {'; '.join(guide.use_cases[:2])}</div>
                      <div style="margin-top:0.35rem"><b>Выход:</b> {'; '.join(guide.outputs[:3])}</div>
                      <div style="margin-top:0.35rem"><b>Ограничения:</b> {'; '.join(guide.limitations[:2])}</div>
                      <div style="margin-top:0.35rem"><b>Отличие:</b> {'; '.join(guide.differs_from[:2])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_upload_intro() -> None:
    """Render tool purpose and data requirements on the upload screen."""
    st.info(
        "Инструмент помогает провести исследование продуктовых изменений: загрузить CSV, "
        "выбрать подходящий метод, получить эффект, диагностики, графики, baseline-анализ "
        "или проверку аномалий."
    )

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("### Panel данные")
            st.caption("Одна строка = один объект в один период.")
            st.markdown("**Цель группы**")
            st.write("Оценивать изменения между treated/control объектами и проверять динамику эффекта.")
            st.markdown("**Подходящие методы**")
            st.write("Diff-in-Diff, Event Study; дополнительно можно запускать диагностику аномалий.")
            st.markdown("**Минимум файла**")
            st.write("- `unit_id`: объект анализа, например user, store, region.")
            st.write("- `date`: дата или период наблюдения.")
            st.write("- `group`: treated/control или аналогичная группа.")
            st.write("- `metric_value`: числовая целевая метрика.")
            st.write("- `treatment_start`: дата воздействия, если она отличается по объектам.")
            st.markdown("**Минимум метрик**")
            st.write("- Одна primary metric для первого анализа.")
            st.write("- Guardrail-метрики можно добавить позже отдельными прогонами.")
            st.write("- Метрика должна быть числовой и сопоставимой между группами.")

    with c2:
        with st.container(border=True):
            st.markdown("### Time series данные")
            st.caption("Одна строка = один период агрегированной метрики.")
            st.markdown("**Цель группы**")
            st.write("Сравнивать факт с ожидаемой динамикой, оценивать эффект события и искать аномалии.")
            st.markdown("**Подходящие методы**")
            st.write("Causal Impact, Interrupted Time Series, Forecast + baseline, Anomaly Detection.")
            st.markdown("**Минимум файла**")
            st.write("- `date`: дата наблюдения.")
            st.write("- `target_metric`: числовая целевая метрика.")
            st.write("- `covariate_*`: опциональные контрольные ряды.")
            st.write("- `intervention_date` можно задать в форме, если нет отдельной колонки.")
            st.write("- Если на дату несколько строк, инструмент агрегирует числовые поля средним.")
            st.markdown("**Минимум метрик**")
            st.write("- Одна target metric для анализа.")
            st.write("- Ковариаты не должны быть следствием интервенции.")
            st.write("- Желателен достаточно длинный pre/baseline период.")

    with st.expander("Общие требования к CSV и ограничения MVP", expanded=False):
        st.write("- Один CSV-файл за один анализ.")
        st.write("- Разделитель: запятая (`,`) или точка с запятой (`;`) — определяется автоматически.")
        st.write("- Кодировка UTF-8 (в т.ч. с BOM) или совместимая с pandas.")
        st.write("- Рекомендуемый размер файла: до 100 MB.")
        st.write("- Жёсткий лимит Streamlit по умолчанию: 200 MB на файл.")
        st.write("- Не загружайте персональные данные, секреты, токены и коммерчески чувствительные выгрузки без обезличивания.")


def _render_dataset_profile(df: pd.DataFrame) -> None:
    """Render describe-like dataset profile."""
    profile = st.session_state["profile"]
    st.write(f"Формат: **{profile.detected_format.value}**")
    st.write(f"Размер: **{profile.n_rows:,}** строк, **{profile.n_cols:,}** колонок")
    st.write(f"Рекомендованная колонка даты: `{profile.suggested_time_col}`")
    st.write(f"Рекомендованные метрики: {', '.join(profile.suggested_metric_cols) or 'не найдены'}")

    rows = []
    for col in profile.columns:
        rows.append(
            {
                "column": col.name,
                "dtype": col.dtype,
                "non_null": profile.n_rows - col.n_missing,
                "missing": col.n_missing,
                "missing_pct": f"{100 * col.n_missing / max(profile.n_rows, 1):.2f}%",
                "unique": col.n_unique,
                "sample": ", ".join(col.sample_values),
            }
        )
    st.markdown("**Описание полей**")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        st.markdown("**Числовая статистика**")
        numeric_stats = df[numeric_cols].describe().transpose().reset_index(names="column")
        st.dataframe(numeric_stats, width="stretch", hide_index=True)

    date_stats = []
    for col in df.columns:
        col_hint = col.lower()
        looks_like_date = any(
            token in col_hint for token in ("date", "time", "day", "period", "week", "month")
        )
        if not looks_like_date and not pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            date_stats.append(
                {
                    "column": col,
                    "min": parsed.min(),
                    "max": parsed.max(),
                    "parsed_pct": f"{100 * parsed.notna().mean():.2f}%",
                }
            )
    if date_stats:
        st.markdown("**Дата/время поля**")
        st.dataframe(pd.DataFrame(date_stats), width="stretch", hide_index=True)


def _render_data_preview(df: pd.DataFrame) -> None:
    """Render preview with selectable columns to avoid hidden right-side columns."""
    st.subheader("Предпросмотр файла")
    default_cols = list(df.columns[: min(12, len(df.columns))])
    selected_cols = st.multiselect(
        "Колонки для предпросмотра",
        options=list(df.columns),
        default=default_cols,
        help="Если колонок много, выберите нужные: так правые колонки не будут скрываться за горизонтальным скроллом.",
    )
    preview_cols = selected_cols or default_cols
    st.dataframe(
        df.loc[:, preview_cols].head(50),
        width="stretch",
        hide_index=True,
        height=420,
    )


def _step_upload() -> None:
    st.header("1. Загрузка данных")
    _render_upload_intro()
    uploaded = st.file_uploader("CSV файл", type=["csv"])
    if uploaded is None:
        st.info("Загрузите panel CSV (unit, date, group, metric) или time series CSV (date, target_metric).")
        return

    try:
        df = load_csv(BytesIO(uploaded.getvalue()))
        st.session_state["df"] = df
        st.session_state["profile"] = profile_dataset(df)
        st.success(f"Загружено {len(df)} строк, {len(df.columns)} колонок.")
        _render_data_preview(df)

        with st.expander("Профиль датасета", expanded=True):
            _render_dataset_profile(df)

        if st.button("Далее → выбор метода"):
            _go_to_step(2)
    except ResearchPlatformError as exc:
        st.error(str(exc))


def _step_method() -> None:
    st.header("2. Метод анализа")
    profile = st.session_state["profile"]
    rec = recommend_method(profile)
    group_methods = _method_group(rec.method, rec.alternatives)
    _render_method_comparison(
        group_methods=group_methods,
        recommended_method=rec.method,
        recommendation_reason=f"{rec.reason} Уверенность: {rec.confidence}.",
    )

    st.divider()
    st.subheader("Выбор метода для запуска")
    ordered_options = _dedupe_methods([*group_methods, *list(AnalysisMethod)])
    method = st.selectbox(
        "Выберите метод",
        options=ordered_options,
        format_func=lambda m: METHOD_LABELS[m],
        index=ordered_options.index(rec.method),
    )
    st.session_state["method"] = method
    with st.expander("Уточняющие примеры use cases для выбранного метода", expanded=True):
        _render_selected_use_cases(method)

    c1, c2 = st.columns([1, 2])
    with c1:
        _back_button("← Назад к загрузке", 1)
    with c2:
        if st.button("Далее → параметры"):
            _go_to_step(3)


def _column_select(
    label: str,
    options: list[str],
    suggested: str | None,
    help_text: str | None = None,
) -> str:
    idx = options.index(suggested) if suggested and suggested in options else 0
    value = st.selectbox(label, options, index=idx, help=help_text)
    if help_text:
        st.caption(help_text)
    return value


def _suggest_event_column(cols: list[str]) -> str | None:
    """Suggest event date column from common naming conventions."""
    for token in ("treatment_start", "event_date", "intervention_date"):
        for col in cols:
            if col.lower() == token:
                return col
    for col in cols:
        lowered = col.lower()
        if "event" in lowered or "treatment" in lowered or "intervention" in lowered:
            return col
    return None


def _default_event_date(df: pd.DataFrame, time_col: str) -> pd.Timestamp:
    """Pick a global event date with both pre and post observations."""
    dates = pd.to_datetime(df[time_col], errors="coerce").dropna()
    if dates.empty:
        return pd.Timestamp.today().normalize()
    unique_dates = sorted(dates.dt.date.unique())
    split_idx = max(min(int(len(unique_dates) * 0.7), len(unique_dates) - 1), 1)
    return pd.Timestamp(unique_dates[split_idx])


def _event_window_defaults(
    df: pd.DataFrame,
    time_col: str,
    event_date_col: str | None,
    global_event_date: pd.Timestamp,
) -> tuple[int, int]:
    """Calculate leads/lags defaults that exist in the data."""
    dates = pd.to_datetime(df[time_col], errors="coerce")
    periods = dates.dt.to_period("M")
    if event_date_col:
        event_dates = pd.to_datetime(df[event_date_col], errors="coerce")
        event_periods = event_dates.dt.to_period("M")
        valid = periods.notna() & event_periods.notna()
        values = [
            (period - event_period).n
            for period, event_period in zip(periods[valid], event_periods[valid])
        ]
        rel_times = pd.Series(values)
    else:
        event_period = pd.Period(global_event_date, freq="M")
        rel_times = (periods - event_period).dropna().apply(lambda value: value.n)
    if rel_times.empty:
        return 6, 6
    n_leads = min(6, max(1, abs(int(rel_times[rel_times < 0].min())) if (rel_times < 0).any() else 1))
    n_lags = min(6, max(1, int(rel_times[rel_times > 0].max()) if (rel_times > 0).any() else 1))
    return n_leads, n_lags


def _default_period_split(df: pd.DataFrame, date_col: str) -> tuple[object, object, object, object]:
    """Return default baseline/pre and analysis/post periods."""
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    dmin, dmax = dates.min().date(), dates.max().date()
    unique_dates = sorted(dates.dt.date.unique())
    split_idx = max(int(len(unique_dates) * 0.7), 1)
    split_idx = min(split_idx, len(unique_dates) - 1) if len(unique_dates) > 1 else 0
    baseline_end = unique_dates[split_idx - 1] if split_idx > 0 else dmin
    analysis_start = unique_dates[split_idx] if split_idx < len(unique_dates) else dmax
    return dmin, baseline_end, analysis_start, dmax


def _step_params() -> None:
    st.header("3. Параметры")
    df: pd.DataFrame = st.session_state["df"]
    profile = st.session_state["profile"]
    cols = list(df.columns)
    method: AnalysisMethod = st.session_state["method"]

    if method == AnalysisMethod.DID:
        unit_col = _column_select(
            "Колонка unit",
            cols,
            profile.suggested_unit_col,
            "Объект анализа: пользователь, магазин, регион, аккаунт или другая единица наблюдения.",
        )
        time_col = _column_select(
            "Колонка даты",
            cols,
            profile.suggested_time_col,
            "Дата или период наблюдения. По ней строится динамика до и после воздействия.",
        )
        group_col = _column_select(
            "Колонка группы",
            cols,
            profile.suggested_group_col,
            "Колонка, которая отделяет объекты с воздействием от контрольной группы.",
        )
        outcome_col = _column_select(
            "Метрика",
            cols,
            profile.suggested_metric_cols[0] if profile.suggested_metric_cols else None,
            "Целевая числовая метрика, эффект на которую нужно оценить.",
        )
        treated_label = st.text_input("Значение treated в группе", value="treated")
        st.caption("Значение в колонке группы, которое означает получение продуктового изменения.")
        treatment_start_col = None
        intervention_date = None
        if profile.has_treatment_start:
            treatment_start_col = _column_select(
                "Колонка даты начала воздействия",
                cols,
                next((c.name for c in profile.columns if "treatment" in c.name.lower() or "event" in c.name.lower()), None),
                "Дата, с которой объект начал получать воздействие. Если дата одна для всех, можно использовать поле ниже.",
            )
        else:
            intervention_date = st.date_input("Дата интервенции (общая для всех treated)")
            st.caption("Общая дата запуска изменения для всех объектов из treated-группы.")

        params = DidParams(
            unit_col=unit_col,
            time_col=time_col,
            group_col=group_col,
            outcome_col=outcome_col,
            treated_label=treated_label,
            intervention_date=intervention_date,
            treatment_start_col=treatment_start_col,
        )
        st.session_state["params"] = params

    elif method == AnalysisMethod.EVENT_STUDY:
        unit_col = _column_select(
            "Колонка unit",
            cols,
            profile.suggested_unit_col,
            "Объект анализа, для которого строится динамика относительно события.",
        )
        time_col = _column_select(
            "Колонка даты",
            cols,
            profile.suggested_time_col,
            "Дата наблюдения. Инструмент переведёт даты в относительные периоды вокруг события.",
        )
        outcome_col = _column_select(
            "Метрика",
            cols,
            profile.suggested_metric_cols[0] if profile.suggested_metric_cols else None,
            "Целевая метрика, динамику эффекта которой нужно оценить.",
        )
        group_col = _column_select(
            "Колонка группы (опционально)",
            ["—"] + cols,
            profile.suggested_group_col,
            "Если есть treated/control, выберите колонку группы. Event Study будет сфокусирован на treated.",
        )
        group_col = None if group_col == "—" else group_col
        suggested_event_col = _suggest_event_column(cols)
        event_col_options = ["—"] + cols
        event_date_col = _column_select(
            "Колонка даты события (если есть)",
            event_col_options,
            suggested_event_col,
            "Дата события для каждого объекта. Если её нет, ниже задаётся одна общая дата события.",
        )
        event_date_col = None if event_date_col == "—" else event_date_col
        default_event_date = _default_event_date(df, time_col)
        global_event_date = None
        if event_date_col:
            st.caption(f"Будет использована дата события из колонки `{event_date_col}`.")
        else:
            global_event_date = st.date_input(
                "Дата события (если одна для всех)",
                value=default_event_date.date(),
            )
            st.caption("Общая дата, относительно которой считаются периоды до и после события.")
        default_leads, default_lags = _event_window_defaults(
            df,
            time_col,
            event_date_col,
            pd.Timestamp(global_event_date or default_event_date),
        )
        with st.expander("Advanced"):
            n_leads = st.number_input("Периодов до события", min_value=1, max_value=24, value=default_leads)
            st.caption("Сколько периодов до события включить в проверку pre-trends.")
            n_lags = st.number_input("Периодов после события", min_value=1, max_value=24, value=default_lags)
            st.caption("Сколько периодов после события включить в оценку динамики эффекта.")
        params = EventStudyParams(
            unit_col=unit_col,
            time_col=time_col,
            outcome_col=outcome_col,
            event_date_col=event_date_col,
            group_col=group_col,
            global_event_date=global_event_date,
            n_leads=int(n_leads),
            n_lags=int(n_lags),
        )
        st.session_state["params"] = params

    elif method == AnalysisMethod.CAUSAL_IMPACT:
        date_col = _column_select(
            "Колонка даты",
            cols,
            profile.suggested_time_col,
            "Дата временного ряда. Для Causal Impact нужна одна агрегированная точка на дату; если строк несколько, инструмент усреднит их.",
        )
        target_col = _column_select(
            "Целевая метрика",
            cols,
            profile.suggested_metric_cols[0] if profile.suggested_metric_cols else None,
            "Метрика, для которой строится counterfactual-прогноз без воздействия.",
        )
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
        covariates = st.multiselect("Ковариаты (необязательно)", numeric)
        st.caption("Контрольные ряды, которые помогают прогнозу, но не должны быть затронуты интервенцией.")
        dmin, default_pre_end, default_post_start, dmax = _default_period_split(df, date_col)
        st.caption(f"Диапазон данных: {dmin} — {dmax}")
        c1, c2 = st.columns(2)
        with c1:
            pre_start = st.date_input("Pre start", value=dmin)
            st.caption("Начало периода до воздействия, на котором обучается counterfactual-модель.")
            pre_end = st.date_input("Pre end", value=default_pre_end)
            st.caption("Конец периода до воздействия. Он должен быть раньше post-period.")
        with c2:
            post_start = st.date_input("Post start", value=default_post_start)
            st.caption("Начало периода после воздействия, где оценивается эффект.")
            post_end = st.date_input("Post end", value=dmax)
            st.caption("Конец периода после воздействия.")
        params = CausalImpactParams(
            date_col=date_col,
            target_col=target_col,
            pre_start=pre_start,
            pre_end=pre_end,
            post_start=post_start,
            post_end=post_end,
            covariate_cols=covariates,
        )
        st.session_state["params"] = params

    elif method == AnalysisMethod.ITS:
        date_col = _column_select(
            "Колонка даты",
            cols,
            profile.suggested_time_col,
            "Дата временного ряда. ITS использует её для построения тренда до и после интервенции.",
        )
        target_col = _column_select(
            "Целевая метрика",
            cols,
            profile.suggested_metric_cols[0] if profile.suggested_metric_cols else None,
            "Метрика, для которой нужно оценить сдвиг уровня и изменение тренда.",
        )
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
        covariates = st.multiselect("Ковариаты (необязательно)", numeric)
        st.caption("Ковариаты помогают учесть сопутствующие факторы, если они не являются следствием интервенции.")
        _, _, default_intervention, _ = _default_period_split(df, date_col)
        intervention_date = st.date_input("Дата интервенции", value=default_intervention)
        st.caption("Дата, после которой метод оценивает изменение уровня и наклона тренда.")
        params = InterruptedTimeSeriesParams(
            date_col=date_col,
            target_col=target_col,
            intervention_date=intervention_date,
            covariate_cols=covariates,
        )
        st.session_state["params"] = params

    elif method == AnalysisMethod.FORECAST_BASELINE:
        date_col = _column_select(
            "Колонка даты",
            cols,
            profile.suggested_time_col,
            "Дата временного ряда для обучения baseline и сравнения с фактом.",
        )
        target_col = _column_select(
            "Целевая метрика",
            cols,
            profile.suggested_metric_cols[0] if profile.suggested_metric_cols else None,
            "Метрика, для которой строится baseline-прогноз.",
        )
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
        covariates = st.multiselect("Ковариаты (необязательно)", numeric)
        st.caption("Ковариаты используются в baseline-прогнозе, если доступны в обоих периодах.")
        baseline_start_default, baseline_end_default, analysis_start_default, analysis_end_default = (
            _default_period_split(df, date_col)
        )
        st.caption(f"Диапазон данных: {baseline_start_default} — {analysis_end_default}")
        c1, c2 = st.columns(2)
        with c1:
            baseline_start = st.date_input("Baseline start", value=baseline_start_default)
            st.caption("Начало исторического периода, на котором обучается baseline.")
            baseline_end = st.date_input("Baseline end", value=baseline_end_default)
            st.caption("Конец исторического периода для обучения baseline.")
        with c2:
            analysis_start = st.date_input("Analysis start", value=analysis_start_default)
            st.caption("Начало периода, где факт сравнивается с baseline.")
            analysis_end = st.date_input("Analysis end", value=analysis_end_default)
            st.caption("Конец периода сравнения факта с baseline.")
        params = ForecastBaselineParams(
            date_col=date_col,
            target_col=target_col,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            covariate_cols=covariates,
        )
        st.session_state["params"] = params

    else:
        date_col = _column_select(
            "Колонка даты",
            cols,
            profile.suggested_time_col,
            "Дата временного ряда для построения expected range и поиска аномалий.",
        )
        target_col = _column_select(
            "Целевая метрика",
            cols,
            profile.suggested_metric_cols[0] if profile.suggested_metric_cols else None,
            "Метрика, в которой нужно найти необычные точки или периоды.",
        )
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
        covariates = st.multiselect("Ковариаты (необязательно)", numeric)
        st.caption("Ковариаты помогают точнее оценить ожидаемый диапазон нормального поведения.")
        baseline_start_default, baseline_end_default, analysis_start_default, analysis_end_default = (
            _default_period_split(df, date_col)
        )
        c1, c2 = st.columns(2)
        with c1:
            baseline_start = st.date_input("Baseline start", value=baseline_start_default)
            st.caption("Период нормального поведения, на котором обучается baseline.")
            baseline_end = st.date_input("Baseline end", value=baseline_end_default)
            st.caption("Конец baseline-period.")
        with c2:
            analysis_start = st.date_input("Analysis start", value=analysis_start_default)
            st.caption("Начало периода поиска аномалий.")
            analysis_end = st.date_input("Analysis end", value=analysis_end_default)
            st.caption("Конец периода поиска аномалий.")
        with st.expander("Advanced"):
            z_threshold = st.number_input("Порог z-score", min_value=1.0, max_value=6.0, value=3.0, step=0.5)
            st.caption("Чем ниже порог, тем больше точек будет отмечено как аномалии.")
        params = AnomalyDetectionParams(
            date_col=date_col,
            target_col=target_col,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            covariate_cols=covariates,
            z_threshold=float(z_threshold),
        )
        st.session_state["params"] = params

    c1, c2 = st.columns([1, 2])
    with c1:
        _back_button("← Назад к выбору метода", 2)
    with c2:
        if st.button("Запустить анализ"):
            try:
                if method == AnalysisMethod.DID:
                    result = run_did(df, st.session_state["params"])
                elif method == AnalysisMethod.EVENT_STUDY:
                    result = run_event_study(df, st.session_state["params"])
                elif method == AnalysisMethod.CAUSAL_IMPACT:
                    result = run_causal_impact(df, st.session_state["params"])
                elif method == AnalysisMethod.ITS:
                    result = run_interrupted_time_series(df, st.session_state["params"])
                elif method == AnalysisMethod.FORECAST_BASELINE:
                    result = run_forecast_baseline(df, st.session_state["params"])
                else:
                    result = run_anomaly_detection(df, st.session_state["params"])
                st.session_state["result"] = result
                _go_to_step(4)
            except ResearchPlatformError as exc:
                st.error(str(exc))


def _step_results() -> None:
    st.header("4. Результаты")
    result = st.session_state["result"]
    c1, c2 = st.columns([1, 4])
    with c1:
        _back_button("← Назад к параметрам", 3)
    with c2:
        st.caption(f"Метод: {METHOD_LABELS[result.method]}")

    st.subheader("Ключевой вывод")
    if result.conclusion:
        st.success(result.conclusion)
    else:
        st.write(result.summary)

    m1, m2, m3, m4 = st.columns(4)
    if result.effect_estimate is not None:
        m1.metric("Эффект", f"{result.effect_estimate:.4f}")
    if result.relative_effect_pct is not None:
        m2.metric("Relative lift", f"{result.relative_effect_pct:.2f}%")
    if result.p_value is not None:
        m3.metric("p-value", format_p_value(result.p_value))
    if result.risk_level:
        m4.metric("Риск", _format_risk(result.risk_level))

    if result.ci_lower is not None and result.ci_upper is not None:
        st.caption(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

    st.subheader("Продуктовая рекомендация")
    st.info(f"💡 {result.recommendation}")
    if result.decision:
        st.caption(f"Решение: {_format_decision(result.decision)}")

    if result.statistical_metrics:
        st.subheader("Статистические метрики")
        metric_rows = [
            {
                "Метрика": metric.name,
                "Значение": metric.value,
                "Описание": metric.description or "Описание метрики: вспомогательный показатель результата анализа.",
            }
            for metric in result.statistical_metrics
        ]
        st.dataframe(
            pd.DataFrame(metric_rows),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Интерпретация")
    st.write(result.interpretation)

    for plot in result.plots:
        st.plotly_chart(plot.figure, width="stretch")

    with st.expander("Предпосылки и диагностики", expanded=True):
        st.markdown("**Предпосылки метода**")
        for a in result.assumptions:
            st.write(f"🔎 {a}")
        st.markdown("**Диагностики качества анализа**")
        for d in result.diagnostics:
            _render_diagnostic(d.name, d.passed, d.message)

    md = result_to_markdown(result)
    st.download_button("Скачать отчёт (Markdown)", md, file_name="analysis_report.md", mime="text/markdown")
    notebook = build_reproduction_notebook(
        df=st.session_state["df"],
        method=result.method,
        params=st.session_state.get("params"),
        result=result,
    )
    st.download_button(
        "Скачать notebook для воспроизведения (.ipynb)",
        notebook,
        file_name="analysis_reproduction.ipynb",
        mime="application/x-ipynb+json",
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Новый анализ"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state["scroll_to_top"] = True
            st.rerun()
    with c2:
        if st.button("Изменить метод"):
            _go_to_step(2)


def run_app() -> None:
    """Console entrypoint: start the Streamlit server for this app.

    Calling ``main()`` directly only works inside the ``streamlit run`` runtime, so the
    ``research-app`` console script launches the server on this file instead.
    """
    import sys
    from pathlib import Path

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    raise SystemExit(stcli.main())


def main() -> None:
    """Entry point for Streamlit app."""
    st.set_page_config(page_title="Research Platform", layout="wide")
    st.title("Research Platform — квази-эксперименты")
    _init_state()
    _scroll_to_top_if_needed()

    step = st.session_state["step"]
    if step == 1:
        _step_upload()
    elif step == 2:
        if st.session_state["df"] is None:
            _go_to_step(1)
        _step_method()
    elif step == 3:
        _step_params()
    else:
        _step_results()


if __name__ == "__main__":
    main()
