"""Causal Impact-style counterfactual model for MVP."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

from research_platform.core.insights import build_product_conclusion, format_p_value
from research_platform.core.schemas import (
    AnalysisMethod,
    AnalysisResult,
    CausalImpactParams,
    DiagnosticItem,
    PlotSpec,
    StatisticalMetric,
)
from research_platform.core.validation import ensure_period_coverage, validate_causal_impact_inputs
from research_platform.utils.errors import EstimationError
from research_platform.visualization.style import BLUE, LIGHT_FILL, ORANGE, apply_plot_style


def run_causal_impact(df: pd.DataFrame, params: CausalImpactParams) -> AnalysisResult:
    """
    Fit trend + optional exogenous regressors on pre-period, forecast post counterfactual.

    This is a simplified Causal Impact-style estimator for MVP. It intentionally uses a stable
    OLS counterfactual rather than a full Bayesian structural time-series model.
    """
    work = validate_causal_impact_inputs(df, params)
    work = work.sort_values(params.date_col).dropna(subset=[params.target_col])
    value_cols = [params.target_col, *params.covariate_cols]
    aggregation = {col: "mean" for col in value_cols if col in work.columns}
    work = work.groupby(params.date_col, as_index=False).agg(aggregation).sort_values(params.date_col)
    dates = work[params.date_col]

    ensure_period_coverage(dates, params.pre_start, params.pre_end, "Pre-period")
    ensure_period_coverage(dates, params.post_start, params.post_end, "Post-period")

    pre_mask = (dates >= pd.Timestamp(params.pre_start)) & (dates <= pd.Timestamp(params.pre_end))
    post_mask = (dates >= pd.Timestamp(params.post_start)) & (dates <= pd.Timestamp(params.post_end))

    pre = work.loc[pre_mask].copy()
    post = work.loc[post_mask].copy()

    if len(pre) < 10:
        raise EstimationError("Causal Impact: pre-period should have at least 10 observations.")

    try:
        covariate_fill_values = {
            col: float(pd.to_numeric(pre[col], errors="coerce").median())
            for col in params.covariate_cols
        }
        pre_features = pd.DataFrame({"trend": np.arange(len(pre), dtype=float)}, index=pre.index)
        post_features = pd.DataFrame(
            {"trend": np.arange(len(pre), len(pre) + len(post), dtype=float)},
            index=post.index,
        )
        for col in params.covariate_cols:
            pre_features[col] = pd.to_numeric(pre[col], errors="coerce").fillna(
                covariate_fill_values[col]
            )
            post_features[col] = pd.to_numeric(post[col], errors="coerce").fillna(
                covariate_fill_values[col]
            )

        x_pre = sm.add_constant(pre_features, has_constant="add")
        x_post = sm.add_constant(post_features, has_constant="add")
        x_post = x_post.reindex(columns=x_pre.columns, fill_value=1.0)
        endog = pre[params.target_col].astype(float)
        res = sm.OLS(endog, x_pre).fit()
        prediction = res.get_prediction(x_post)
        pred_mean = np.asarray(prediction.predicted_mean, dtype=float)
        pred_frame = prediction.summary_frame(alpha=0.05)
        pred_lower = pred_frame["mean_ci_lower"].to_numpy(dtype=float)
        pred_upper = pred_frame["mean_ci_upper"].to_numpy(dtype=float)
    except Exception as exc:
        raise EstimationError(f"Causal Impact model failed: {exc}") from exc

    if len(pred_mean) == 0 or np.isnan(pred_mean).all():
        raise EstimationError("Causal Impact model produced empty counterfactual forecast.")

    observed = post[params.target_col].astype(float).values
    pointwise = observed - pred_mean
    cumulative = float(pointwise.sum())
    avg_effect = float(pointwise.mean())
    predicted_avg = float(np.mean(pred_mean))
    relative_effect_pct = 100.0 * avg_effect / predicted_avg if predicted_avg else None

    full_dates = work.loc[pre_mask | post_mask, params.date_col]
    full_observed = work.loc[pre_mask | post_mask, params.target_col]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=full_dates,
            y=full_observed,
            mode="lines",
            name="Факт",
            line=dict(color=BLUE, width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=post[params.date_col],
            y=pred_mean,
            mode="lines",
            name="Прогноз без воздействия",
            line=dict(color=ORANGE, width=2.4, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(post[params.date_col]) + list(post[params.date_col][::-1]),
            y=list(pred_upper) + list(pred_lower[::-1]),
            fill="toself",
            fillcolor=LIGHT_FILL,
            line=dict(color="rgba(0,0,0,0)"),
            name="95% интервал прогноза",
            hoverinfo="skip",
        )
    )
    fig.add_vline(x=pd.Timestamp(params.post_start), line_dash="dash", line_color=ORANGE)
    fig = apply_plot_style(
        fig,
        title="Causal Impact: факт и counterfactual-прогноз",
        xaxis_title="Дата",
        yaxis_title=params.target_col,
    )

    impact_fig = go.Figure()
    impact_fig.add_trace(
        go.Bar(x=post[params.date_col], y=pointwise, name="Pointwise impact", marker_color=BLUE)
    )
    impact_fig.add_hline(y=0, line_dash="dash", line_color=ORANGE)
    impact_fig = apply_plot_style(
        impact_fig,
        title="Pointwise impact по периодам",
        xaxis_title="Дата",
        yaxis_title=f"{params.target_col}: observed - counterfactual",
    )

    cumulative_effects = np.cumsum(pointwise)
    cumulative_fig = go.Figure()
    cumulative_fig.add_trace(
        go.Scatter(
            x=post[params.date_col],
            y=cumulative_effects,
            mode="lines+markers",
            name="Накопленный эффект",
            line=dict(color=BLUE, width=2.5),
            marker=dict(color=BLUE, size=6),
        )
    )
    cumulative_fig.add_hline(y=0, line_dash="dash", line_color=ORANGE)
    cumulative_fig = apply_plot_style(
        cumulative_fig,
        title="Накопленный эффект",
        xaxis_title="Дата",
        yaxis_title="Cumulative impact",
    )

    se = float(np.std(pointwise, ddof=1) / np.sqrt(len(pointwise))) if len(pointwise) > 1 else 0.0
    ci_lo = avg_effect - 1.96 * se
    ci_hi = avg_effect + 1.96 * se
    z_stat = avg_effect / se if se else np.nan
    p_value = math.erfc(abs(z_stat) / math.sqrt(2)) if not np.isnan(z_stat) else None
    conclusion, recommendation, risk_level = build_product_conclusion(
        metric_name=params.target_col,
        effect=avg_effect,
        relative_effect_pct=relative_effect_pct,
        p_value=p_value,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        method_label="Causal Impact",
    )

    relative_text = f"{relative_effect_pct:.2f}%" if relative_effect_pct is not None else "n/a"

    return AnalysisResult(
        method=AnalysisMethod.CAUSAL_IMPACT,
        summary=f"Средний pointwise эффект: {avg_effect:.4f}; cumulative: {cumulative:.4f}",
        effect_estimate=avg_effect,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p_value=p_value,
        relative_effect_pct=relative_effect_pct,
        assumptions=[
            "Стабильная pre-period динамика для прогноза counterfactual.",
            "Ковариаты не затронуты интервенцией.",
            "Нет существенных структурных сдвигов в post, кроме интервенции.",
        ],
        diagnostics=[
            DiagnosticItem(
                name="pre_period_length",
                passed=len(pre) >= 10,
                message=f"Pre-period наблюдений: {len(pre)}.",
            ),
            DiagnosticItem(
                name="time_series_aggregation",
                passed=True,
                message=(
                    "Если в данных было несколько строк на одну дату, инструмент агрегировал "
                    "целевую метрику и ковариаты средним значением по дате."
                ),
            )
        ],
        statistical_metrics=[
            StatisticalMetric(
                name="Pre observations",
                value=str(len(pre)),
                description="Количество наблюдений для обучения counterfactual-модели",
            ),
            StatisticalMetric(
                name="Post observations",
                value=str(len(post)),
                description="Количество наблюдений после интервенции",
            ),
            StatisticalMetric(
                name="Observed post mean",
                value=f"{float(np.mean(observed)):.4f}",
                description="Среднее фактическое значение целевой метрики в post-period",
            ),
            StatisticalMetric(
                name="Counterfactual post mean",
                value=f"{predicted_avg:.4f}",
                description="Средний прогноз метрики в post-period без воздействия",
            ),
            StatisticalMetric(
                name="Avg impact",
                value=f"{avg_effect:.4f}",
                description="Среднее отклонение observed от counterfactual в post-period",
            ),
            StatisticalMetric(
                name="Cumulative impact",
                value=f"{cumulative:.4f}",
                description="Суммарный эффект за весь post-period",
            ),
            StatisticalMetric(
                name="Relative impact",
                value=f"{relative_effect_pct:.2f}%" if relative_effect_pct is not None else "n/a",
                description="Средний эффект относительно среднего counterfactual",
            ),
            StatisticalMetric(
                name="Approx. p-value",
                value=format_p_value(p_value),
                description="Приближённая значимость среднего pointwise impact",
            ),
        ],
        conclusion=conclusion,
        interpretation=(
            f"В post-period наблюдаемое отклонение от counterfactual в среднем {avg_effect:.4f} "
            f"(cumulative {cumulative:.4f}, relative {relative_text})."
        ),
        recommendation=recommendation,
        decision="scale" if p_value is not None and p_value < 0.05 and avg_effect > 0 else "hold_or_iterate",
        risk_level=risk_level,
        plots=[
            PlotSpec(title="Observed vs counterfactual", figure=fig),
            PlotSpec(title="Pointwise impact", figure=impact_fig),
            PlotSpec(title="Cumulative impact", figure=cumulative_fig),
        ],
        extra={
            "pointwise_effects": pointwise.tolist(),
            "cumulative_effect": cumulative,
            "predicted_mean": pred_mean.tolist(),
            "observed": observed.tolist(),
        },
    )
