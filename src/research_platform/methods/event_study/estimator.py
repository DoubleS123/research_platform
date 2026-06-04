"""Event Study with relative time dummies."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf

from research_platform.core.insights import build_product_conclusion, format_p_value
from research_platform.core.schemas import (
    AnalysisMethod,
    AnalysisResult,
    DiagnosticItem,
    EventStudyParams,
    PlotSpec,
    StatisticalMetric,
)
from research_platform.core.validation import validate_event_study_inputs
from research_platform.utils.errors import EstimationError
from research_platform.visualization.style import BLUE, LIGHT_BLUE, ORANGE, apply_plot_style


def _assign_relative_periods(work: pd.DataFrame, params: EventStudyParams) -> pd.DataFrame:
    """Map calendar dates to integer periods relative to the event."""
    periods = work[params.time_col].dt.to_period("M")
    if params.event_date_col and params.event_date_col in work.columns:
        event_period = work.groupby(params.unit_col)[params.event_date_col].transform("min").dt.to_period("M")
    elif params.global_event_date:
        event_period = pd.Period(params.global_event_date, freq="M")
    else:
        raise EstimationError("Event Study: event date required.")
    work = work.copy()
    if isinstance(event_period, pd.Period):
        work["rel_time"] = (periods - event_period).apply(lambda value: value.n)
    else:
        valid = periods.notna() & event_period.notna()
        work["rel_time"] = np.nan
        work.loc[valid, "rel_time"] = [
            (period - event).n for period, event in zip(periods[valid], event_period[valid])
        ]
    return work


def _choose_baseline_period(rel_times: pd.Series) -> tuple[int, bool]:
    """Choose a reference period that exists in the filtered data."""
    unique_periods = sorted(int(period) for period in rel_times.dropna().unique())
    if not unique_periods:
        raise EstimationError("Event Study: no valid relative periods after filtering.")
    if -1 in unique_periods:
        return -1, True
    pre_periods = [period for period in unique_periods if period < 0]
    if pre_periods:
        return max(pre_periods), False
    return min(unique_periods), False


def run_event_study(df: pd.DataFrame, params: EventStudyParams) -> AnalysisResult:
    """Estimate dynamic treatment effects by relative period."""
    work = validate_event_study_inputs(df, params)
    work = work.dropna(subset=[params.outcome_col])
    if params.group_col:
        work = work[work[params.group_col].astype(str).str.lower() == params.treated_label.lower()]
    work = _assign_relative_periods(work, params)
    work = work.dropna(subset=["rel_time"])
    work["rel_time"] = work["rel_time"].astype(int)
    work = work[(work["rel_time"] >= -params.n_leads) & (work["rel_time"] <= params.n_lags)]

    baseline_period, used_default_baseline = _choose_baseline_period(work["rel_time"])

    formula = (
        f"{params.outcome_col} ~ C(rel_time, Treatment(reference={baseline_period})) "
        f"+ C({params.unit_col})"
    )
    try:
        model = smf.ols(formula, data=work).fit(
            cov_type="cluster",
            cov_kwds={"groups": work[params.unit_col]},
        )
    except Exception as exc:
        raise EstimationError(f"Event Study model failed: {exc}") from exc

    coefs: list[float] = []
    ses: list[float] = []
    pvals: list[float] = []
    periods: list[int] = []
    for name, val in model.params.items():
        if "rel_time" in str(name) and "T." in str(name):
            try:
                period = int(str(name).split("T.")[1].rstrip("]"))
            except ValueError:
                continue
            periods.append(period)
            coefs.append(float(val))
            ses.append(float(model.bse.get(name, np.nan)))
            pvals.append(float(model.pvalues.get(name, np.nan)))
    order = np.argsort(periods)
    periods = [periods[i] for i in order]
    coefs = [coefs[i] for i in order]
    ses = [ses[i] for i in order]
    pvals = [pvals[i] for i in order]

    pre_coefs = [c for p, c in zip(periods, coefs) if p < 0]
    pre_ses = [s for p, s in zip(periods, ses) if p < 0]
    pre_passed = all(abs(c) < 2 * s for c, s in zip(pre_coefs, pre_ses)) if pre_coefs else True

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=coefs,
            mode="lines+markers",
            name="coefficient",
            error_y=dict(type="data", array=[1.96 * s for s in ses], visible=True),
            line=dict(color=BLUE, width=2.5),
            marker=dict(color=BLUE, size=7),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=ORANGE)
    fig.add_vline(x=baseline_period, line_dash="dot", line_color=LIGHT_BLUE)
    fig = apply_plot_style(
        fig,
        title="Event Study: коэффициенты по относительному времени",
        xaxis_title="Период относительно события (0 = месяц события)",
        yaxis_title="Эффект",
    )

    distribution = go.Figure()
    for rel_time, sub in work.groupby("rel_time"):
        distribution.add_trace(
            go.Box(
                y=sub[params.outcome_col],
                name=str(rel_time),
                boxpoints="outliers",
                marker_color=BLUE if rel_time >= 0 else ORANGE,
                line_color=BLUE if rel_time >= 0 else ORANGE,
            )
        )
    distribution = apply_plot_style(
        distribution,
        title="Распределение метрики по относительным периодам",
        xaxis_title="Период относительно события",
        yaxis_title=params.outcome_col,
    )

    post_coefs = [c for p, c in zip(periods, coefs) if p >= 0]
    post_pvals = [pval for p, pval in zip(periods, pvals) if p >= 0]
    avg_post = float(np.mean(post_coefs)) if post_coefs else None
    post_se = float(np.std(post_coefs, ddof=1) / np.sqrt(len(post_coefs))) if len(post_coefs) > 1 else None
    ci_lower = avg_post - 1.96 * post_se if avg_post is not None and post_se is not None else None
    ci_upper = avg_post + 1.96 * post_se if avg_post is not None and post_se is not None else None
    p_value = min(post_pvals) if post_pvals else None
    pre_mean = float(work.loc[work["rel_time"] < 0, params.outcome_col].mean())
    rel_pct = 100.0 * avg_post / pre_mean if avg_post is not None and pre_mean else None
    conclusion, recommendation, risk_level = build_product_conclusion(
        metric_name=params.outcome_col,
        effect=avg_post,
        relative_effect_pct=rel_pct,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method_label="Event Study",
    )

    return AnalysisResult(
        method=AnalysisMethod.EVENT_STUDY,
        summary="Event Study: динамика эффекта вокруг события",
        effect_estimate=avg_post,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        relative_effect_pct=rel_pct,
        assumptions=[
            "Параллельные тренды: pre-period коэффициенты близки к нулю.",
            "Нет anticipation effects до события.",
        ],
        diagnostics=[
            DiagnosticItem(
                name="pre_trend_coefficients",
                passed=pre_passed,
                message="Pre-period коэффициенты должны быть статистически близки к нулю.",
            ),
            DiagnosticItem(
                name="baseline_period",
                passed=baseline_period < 0,
                message=(
                    f"Baseline period: {baseline_period}"
                    + (
                        " (период -1)."
                        if used_default_baseline
                        else " — периода -1 нет, взят ближайший доступный pre-period."
                    )
                    + (
                        ""
                        if baseline_period < 0
                        else " Внимание: pre-периодов нет, baseline выбран из post — "
                        "causal-интерпретация ненадёжна."
                    )
                ),
            ),
        ],
        statistical_metrics=[
            StatisticalMetric(
                name="N observations",
                value=str(len(work)),
                description="Количество строк, использованных в модели Event Study",
            ),
            StatisticalMetric(
                name="N units",
                value=str(work[params.unit_col].nunique()),
                description="Количество уникальных объектов анализа",
            ),
            StatisticalMetric(
                name="Baseline period",
                value=str(baseline_period),
                description="Период, относительно которого считаются коэффициенты",
            ),
            StatisticalMetric(
                name="Avg post effect",
                value=f"{avg_post:.4f}" if avg_post is not None else "n/a",
                description="Средний коэффициент после события",
            ),
            StatisticalMetric(
                name="Relative post effect",
                value=f"{rel_pct:.2f}%" if rel_pct is not None else "n/a",
                description="Средний post-period эффект относительно среднего значения метрики до события",
            ),
            StatisticalMetric(
                name="Min post p-value",
                value=format_p_value(p_value),
                description="Минимальный p-value среди post-period коэффициентов",
            ),
        ],
        conclusion=conclusion,
        interpretation=(
            "Смотрите график коэффициентов: он показывает onset, persistence и decay эффекта. "
            "Если pre-period коэффициенты заметно отличаются от нуля, causal-интерпретация слабее."
        ),
        recommendation=recommendation,
        decision="scale" if p_value is not None and p_value < 0.05 and avg_post and avg_post > 0 else "hold_or_iterate",
        risk_level=risk_level,
        plots=[
            PlotSpec(title="Event study coefficients", figure=fig),
            PlotSpec(title="Outcome by relative period", figure=distribution),
        ],
        extra={
            "periods": periods,
            "coefficients": coefs,
            "standard_errors": ses,
            "p_values": pvals,
            "baseline_period": baseline_period,
        },
    )
