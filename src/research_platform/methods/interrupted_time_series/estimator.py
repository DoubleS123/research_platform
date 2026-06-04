"""Interrupted Time Series estimation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

from research_platform.core.insights import build_product_conclusion, format_p_value
from research_platform.core.schemas import (
    AnalysisMethod,
    AnalysisResult,
    DiagnosticItem,
    InterruptedTimeSeriesParams,
    PlotSpec,
    StatisticalMetric,
)
from research_platform.methods.time_series.common import prepare_time_series, relative_effect_pct
from research_platform.utils.errors import EstimationError
from research_platform.visualization.style import BLUE, ORANGE, apply_plot_style


def run_interrupted_time_series(
    df: pd.DataFrame,
    params: InterruptedTimeSeriesParams,
) -> AnalysisResult:
    """Estimate level and trend changes after an intervention."""
    work = prepare_time_series(
        df,
        date_col=params.date_col,
        target_col=params.target_col,
        covariate_cols=params.covariate_cols,
    )
    intervention_ts = pd.Timestamp(params.intervention_date)
    work["post"] = (work[params.date_col] >= intervention_ts).astype(int)
    if work["post"].nunique() < 2:
        raise EstimationError("ITS requires observations before and after intervention_date.")
    first_post_trend = float(work.loc[work["post"] == 1, "trend"].min())
    work["trend_after"] = np.where(work["post"] == 1, work["trend"] - first_post_trend + 1, 0.0)

    feature_cols = ["trend", "post", "trend_after", *params.covariate_cols]
    x = sm.add_constant(work[feature_cols], has_constant="add")
    y = work[params.target_col].astype(float)
    try:
        model = sm.OLS(y, x).fit(cov_type="HC1")
    except Exception as exc:
        raise EstimationError(f"ITS model failed: {exc}") from exc

    work["fitted"] = model.predict(x)
    counterfactual = work.copy()
    counterfactual["post"] = 0
    counterfactual["trend_after"] = 0.0
    x_cf = sm.add_constant(counterfactual[feature_cols], has_constant="add")
    x_cf = x_cf.reindex(columns=x.columns, fill_value=1.0)
    work["counterfactual"] = model.predict(x_cf)

    post = work[work["post"] == 1]
    pointwise = post[params.target_col].to_numpy(dtype=float) - post["counterfactual"].to_numpy(dtype=float)
    avg_effect = float(np.mean(pointwise))
    baseline_mean = float(post["counterfactual"].mean())
    rel_pct = relative_effect_pct(avg_effect, baseline_mean)
    se = float(np.std(pointwise, ddof=1) / np.sqrt(len(pointwise))) if len(pointwise) > 1 else 0.0
    ci_lower = avg_effect - 1.96 * se
    ci_upper = avg_effect + 1.96 * se
    p_value = float(model.pvalues.get("post", np.nan))
    level_change = float(model.params.get("post", np.nan))
    trend_change = float(model.params.get("trend_after", np.nan))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work[params.date_col],
            y=work[params.target_col],
            mode="lines",
            name="Факт",
            line=dict(color=BLUE, width=2.3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=work[params.date_col],
            y=work["fitted"],
            mode="lines",
            name="Segmented fit",
            line=dict(color=ORANGE, width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=post[params.date_col],
            y=post["counterfactual"],
            mode="lines",
            name="Counterfactual без изменения",
            line=dict(color=ORANGE, width=2.0, dash="dash"),
        )
    )
    fig.add_vline(x=intervention_ts, line_dash="dash", line_color=ORANGE)
    fig = apply_plot_style(
        fig,
        title="Interrupted Time Series: уровень и тренд после воздействия",
        xaxis_title="Дата",
        yaxis_title=params.target_col,
    )

    conclusion, recommendation, risk_level = build_product_conclusion(
        metric_name=params.target_col,
        effect=avg_effect,
        relative_effect_pct=rel_pct,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method_label="Interrupted Time Series",
    )

    return AnalysisResult(
        method=AnalysisMethod.ITS,
        summary=f"ITS: level change {level_change:.4f}, trend change {trend_change:.4f}",
        effect_estimate=avg_effect,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        relative_effect_pct=rel_pct,
        assumptions=[
            "До интервенции временной ряд достаточно стабилен, чтобы оценить базовый тренд.",
            "В момент интервенции не произошло других крупных событий, влияющих на метрику.",
            "Изменение после даты интервенции можно описать сдвигом уровня и/или наклона тренда.",
        ],
        diagnostics=[
            DiagnosticItem(
                name="pre_post_coverage",
                passed=work["post"].nunique() == 2,
                message="В данных есть наблюдения до и после даты интервенции.",
            ),
            DiagnosticItem(
                name="level_change_significance",
                passed=bool(p_value < 0.05),
                message=f"p-value для изменения уровня: {format_p_value(p_value)}.",
            ),
        ],
        statistical_metrics=[
            StatisticalMetric(
                name="Level change",
                value=f"{level_change:.4f}",
                description="Мгновенное изменение уровня метрики после даты интервенции",
            ),
            StatisticalMetric(
                name="Trend change",
                value=f"{trend_change:.4f}",
                description="Изменение наклона тренда после даты интервенции",
            ),
            StatisticalMetric(
                name="Avg post effect",
                value=f"{avg_effect:.4f}",
                description="Среднее отклонение факта от counterfactual в post-period",
            ),
            StatisticalMetric(
                name="Relative effect",
                value=f"{rel_pct:.2f}%" if rel_pct is not None else "n/a",
                description="Средний эффект относительно counterfactual",
            ),
            StatisticalMetric(
                name="p-value",
                value=format_p_value(p_value),
                description="Значимость мгновенного изменения уровня после интервенции",
            ),
        ],
        conclusion=conclusion,
        interpretation=(
            "ITS разделяет эффект на два компонента: скачок уровня сразу после события и "
            "изменение наклона тренда. Это удобно, когда нужно понять не только факт изменения, "
            "но и изменилась ли дальнейшая динамика метрики."
        ),
        recommendation=recommendation,
        decision="scale" if p_value < 0.05 and avg_effect > 0 else "hold_or_iterate",
        risk_level=risk_level,
        plots=[PlotSpec(title="Interrupted Time Series", figure=fig)],
        extra={
            "formula_features": feature_cols,
            "level_change": level_change,
            "trend_change": trend_change,
        },
    )
