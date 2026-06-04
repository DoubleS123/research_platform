"""Diff-in-Diff estimation with two-way fixed effects."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf

from research_platform.core.insights import build_product_conclusion, format_p_value
from research_platform.core.schemas import (
    AnalysisMethod,
    AnalysisResult,
    DidParams,
    DiagnosticItem,
    PlotSpec,
    StatisticalMetric,
)
from research_platform.core.validation import validate_did_inputs
from research_platform.utils.errors import EstimationError
from research_platform.visualization.style import BLUE, LIGHT_BLUE, ORANGE, apply_plot_style


def _build_post_indicator(
    df: pd.DataFrame,
    params: DidParams,
) -> pd.Series:
    """Create post-treatment indicator per row."""
    if params.treatment_start_col:
        start = df[params.treatment_start_col]
        return df[params.time_col] >= start
    assert params.intervention_date is not None
    return df[params.time_col] >= pd.Timestamp(params.intervention_date)


def run_did(df: pd.DataFrame, params: DidParams) -> AnalysisResult:
    """
    Estimate DiD via TWFE: Y ~ treated * post + unit FE + time FE.

    Clustered SE at unit level (cluster-robust on unit_id).
    """
    work = validate_did_inputs(df, params)
    work = work.dropna(subset=[params.outcome_col, params.group_col])
    work["treated"] = (work[params.group_col].astype(str).str.lower() == params.treated_label.lower()).astype(int)
    work["post"] = _build_post_indicator(work, params).astype(int)
    work["did"] = work["treated"] * work["post"]

    if work["treated"].nunique() < 2:
        raise EstimationError("DiD requires both treated and control groups.")

    formula = f"{params.outcome_col} ~ did + C({params.unit_col}) + C({params.time_col})"
    try:
        model = smf.ols(formula, data=work).fit(
            cov_type="cluster",
            cov_kwds={"groups": work[params.unit_col]},
        )
    except Exception as exc:
        raise EstimationError(f"DiD model failed: {exc}") from exc

    effect = float(model.params.get("did", np.nan))
    ci = model.conf_int().loc["did"] if "did" in model.params else (np.nan, np.nan)
    pval = float(model.pvalues.get("did", np.nan))

    pre = work[work["post"] == 0].groupby("treated")[params.outcome_col].mean()
    rel_pct = None
    if 0 in pre.index and effect is not None and pre[0] != 0:
        rel_pct = 100.0 * effect / pre[0]

    trend = (
        work.groupby([params.time_col, params.group_col], as_index=False)[params.outcome_col]
        .mean()
        .sort_values(params.time_col)
    )
    fig = go.Figure()
    colors = [BLUE, ORANGE, LIGHT_BLUE]
    for idx, (grp, sub) in enumerate(trend.groupby(params.group_col)):
        fig.add_trace(
            go.Scatter(
                x=sub[params.time_col],
                y=sub[params.outcome_col],
                mode="lines+markers",
                name=str(grp),
                line=dict(color=colors[idx % len(colors)], width=2.5),
                marker=dict(size=6),
            )
        )
    fig = apply_plot_style(
        fig,
        title="Средние значения метрики по группам",
        xaxis_title="Дата",
        yaxis_title=params.outcome_col,
    )

    distribution = go.Figure()
    for (treated, post), sub in work.groupby(["treated", "post"]):
        group_name = "treated" if treated == 1 else "control"
        period_name = "post" if post == 1 else "pre"
        distribution.add_trace(
            go.Box(
                y=sub[params.outcome_col],
                name=f"{group_name} / {period_name}",
                boxpoints="outliers",
                marker_color=BLUE if treated == 1 else ORANGE,
                line_color=BLUE if treated == 1 else ORANGE,
            )
        )
    distribution = apply_plot_style(
        distribution,
        title="Распределение метрики по группам и периодам",
        xaxis_title="Группа / период",
        yaxis_title=params.outcome_col,
    )

    means = (
        work.groupby(["treated", "post"], as_index=False)[params.outcome_col]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    means["group_period"] = means.apply(
        lambda r: f"{'treated' if r['treated'] == 1 else 'control'} / {'post' if r['post'] == 1 else 'pre'}",
        axis=1,
    )
    means_fig = go.Figure(
        go.Bar(
            x=means["group_period"],
            y=means["mean"],
            text=means["mean"].round(2),
            textposition="auto",
            marker_color=[ORANGE if row["treated"] == 0 else BLUE for _, row in means.iterrows()],
        )
    )
    means_fig = apply_plot_style(
        means_fig,
        title="Средние значения по группам и периодам",
        xaxis_title="Группа / период",
        yaxis_title=params.outcome_col,
    )

    diagnostics = [
        DiagnosticItem(
            name="parallel_trends_visual",
            passed=True,
            message="Проверьте график: тренды treated/control должны быть параллельны до интервенции.",
        ),
        DiagnosticItem(
            name="sample_balance",
            passed=work["treated"].sum() > 0 and (work["treated"] == 0).sum() > 0,
            message="В выборке есть treated и control.",
        ),
    ]

    ci_lower = float(ci[0]) if hasattr(ci, "__len__") else None
    ci_upper = float(ci[1]) if hasattr(ci, "__len__") else None
    conclusion, recommendation, risk_level = build_product_conclusion(
        metric_name=params.outcome_col,
        effect=effect,
        relative_effect_pct=rel_pct,
        p_value=pval,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method_label="Diff-in-Diff",
    )
    interpretation = (
        f"Средний эффект воздействия: {effect:.4f} "
        f"(95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], p-value: {format_p_value(pval)})."
    )
    if rel_pct is not None:
        interpretation += f" Относительно среднего control в pre-period: {rel_pct:.1f}%."

    return AnalysisResult(
        method=AnalysisMethod.DID,
        summary=f"Оценка DiD (TWFE): {effect:.4f}",
        effect_estimate=effect,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=pval,
        relative_effect_pct=rel_pct,
        assumptions=[
            "Параллельные тренды в отсутствие воздействия.",
            "Нет существенных одновременных шоков только для одной группы.",
            "Стабильный состав treated/control.",
        ],
        diagnostics=diagnostics,
        statistical_metrics=[
            StatisticalMetric(name="N observations", value=str(len(work)), description="Строк в модели"),
            StatisticalMetric(name="N units", value=str(work[params.unit_col].nunique()), description="Объектов анализа"),
            StatisticalMetric(name="ATE", value=f"{effect:.4f}", description="Средний эффект воздействия по модели Diff-in-Diff"),
            StatisticalMetric(name="95% CI", value=f"[{ci_lower:.4f}, {ci_upper:.4f}]", description="Доверительный интервал"),
            StatisticalMetric(name="p-value", value=format_p_value(pval), description="Вероятность получить такой эффект или сильнее при нулевой гипотезе"),
            StatisticalMetric(name="Relative lift", value=f"{rel_pct:.2f}%" if rel_pct is not None else "n/a", description="Эффект относительно среднего control в pre-period"),
        ],
        conclusion=conclusion,
        interpretation=interpretation,
        recommendation=recommendation,
        decision="scale" if pval < 0.05 and effect > 0 else "hold_or_iterate",
        risk_level=risk_level,
        plots=[
            PlotSpec(title="Trends by group", figure=fig),
            PlotSpec(title="Outcome distribution", figure=distribution),
            PlotSpec(title="Group-period means", figure=means_fig),
        ],
        extra={"formula": formula, "n_obs": len(work), "group_period_means": means.to_dict("records")},
    )
