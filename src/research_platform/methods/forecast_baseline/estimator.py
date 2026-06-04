"""Forecast + baseline analysis."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from research_platform.core.insights import build_product_conclusion, format_p_value
from research_platform.core.schemas import (
    AnalysisMethod,
    AnalysisResult,
    DiagnosticItem,
    ForecastBaselineParams,
    PlotSpec,
    StatisticalMetric,
)
from research_platform.methods.time_series.common import (
    fit_baseline_model,
    period_mask,
    prepare_time_series,
    relative_effect_pct,
)
from research_platform.visualization.style import BLUE, LIGHT_FILL, ORANGE, apply_plot_style


def run_forecast_baseline(df: pd.DataFrame, params: ForecastBaselineParams) -> AnalysisResult:
    """Fit baseline on historical period and compare forecast to actuals."""
    work = prepare_time_series(
        df,
        date_col=params.date_col,
        target_col=params.target_col,
        covariate_cols=params.covariate_cols,
    )
    baseline = work[period_mask(work[params.date_col], params.baseline_start, params.baseline_end)]
    analysis = work[period_mask(work[params.date_col], params.analysis_start, params.analysis_end)]
    prediction = fit_baseline_model(
        train_df=baseline,
        predict_df=analysis,
        target_col=params.target_col,
        covariate_cols=params.covariate_cols,
    )

    observed = analysis[params.target_col].to_numpy(dtype=float)
    delta = observed - prediction.mean
    avg_delta = float(np.mean(delta))
    cumulative_delta = float(np.sum(delta))
    predicted_avg = float(np.mean(prediction.mean))
    rel_pct = relative_effect_pct(avg_delta, predicted_avg)
    se = float(np.std(delta, ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0
    ci_lower = avg_delta - 1.96 * se
    ci_upper = avg_delta + 1.96 * se
    z_stat = avg_delta / se if se else np.nan
    p_value = math.erfc(abs(z_stat) / math.sqrt(2)) if not np.isnan(z_stat) else None

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
            x=analysis[params.date_col],
            y=prediction.mean,
            mode="lines",
            name="Baseline forecast",
            line=dict(color=ORANGE, width=2.3, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(analysis[params.date_col]) + list(analysis[params.date_col][::-1]),
            y=list(prediction.upper) + list(prediction.lower[::-1]),
            fill="toself",
            fillcolor=LIGHT_FILL,
            line=dict(color="rgba(0,0,0,0)"),
            name="95% интервал baseline",
            hoverinfo="skip",
        )
    )
    fig.add_vline(x=pd.Timestamp(params.analysis_start), line_dash="dash", line_color=ORANGE)
    fig = apply_plot_style(
        fig,
        title="Forecast + baseline: факт против ожидаемого baseline",
        xaxis_title="Дата",
        yaxis_title=params.target_col,
    )

    delta_fig = go.Figure()
    delta_fig.add_trace(
        go.Bar(x=analysis[params.date_col], y=delta, name="Факт - baseline", marker_color=BLUE)
    )
    delta_fig.add_hline(y=0, line_dash="dash", line_color=ORANGE)
    delta_fig = apply_plot_style(
        delta_fig,
        title="Отклонение факта от baseline по периодам",
        xaxis_title="Дата",
        yaxis_title="Отклонение",
    )

    conclusion, recommendation, risk_level = build_product_conclusion(
        metric_name=params.target_col,
        effect=avg_delta,
        relative_effect_pct=rel_pct,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method_label="Forecast + baseline",
    )
    recommendation = (
        f"{recommendation} Forecast + baseline показывает отклонение от ожидаемой динамики, "
        "но сам по себе не доказывает причинность без проверки внешних факторов."
    )

    return AnalysisResult(
        method=AnalysisMethod.FORECAST_BASELINE,
        summary=f"Среднее отклонение от baseline: {avg_delta:.4f}; cumulative: {cumulative_delta:.4f}",
        effect_estimate=avg_delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        relative_effect_pct=rel_pct,
        assumptions=[
            "Baseline-period достаточно репрезентативен для прогноза ожидаемой динамики.",
            "В analysis-period нет неучтённых внешних шоков, которые полностью объясняют отклонение.",
            "Ковариаты, если выбраны, доступны и сопоставимы в baseline и analysis периодах.",
        ],
        diagnostics=[
            DiagnosticItem(
                name="baseline_observations",
                passed=len(baseline) >= 10,
                message=f"В baseline-period использовано наблюдений: {len(baseline)}.",
            ),
            DiagnosticItem(
                name="analysis_observations",
                passed=len(analysis) > 0,
                message=f"В analysis-period использовано наблюдений: {len(analysis)}.",
            ),
        ],
        statistical_metrics=[
            StatisticalMetric(
                name="Baseline observations",
                value=str(len(baseline)),
                description="Количество наблюдений, на которых обучался baseline-прогноз",
            ),
            StatisticalMetric(
                name="Analysis observations",
                value=str(len(analysis)),
                description="Количество наблюдений, где факт сравнивался с baseline",
            ),
            StatisticalMetric(
                name="Avg delta",
                value=f"{avg_delta:.4f}",
                description="Среднее отклонение факта от baseline",
            ),
            StatisticalMetric(
                name="Cumulative delta",
                value=f"{cumulative_delta:.4f}",
                description="Суммарное отклонение факта от baseline",
            ),
            StatisticalMetric(
                name="Relative delta",
                value=f"{rel_pct:.2f}%" if rel_pct is not None else "n/a",
                description="Среднее отклонение относительно среднего baseline",
            ),
            StatisticalMetric(
                name="Approx. p-value",
                value=format_p_value(p_value),
                description="Приближённая значимость среднего отклонения от baseline",
            ),
        ],
        conclusion=conclusion,
        interpretation=(
            "Метод отвечает на вопрос: насколько факт отличается от ожидаемого baseline-прогноза. "
            "Он полезен для plan/fact и мониторинга, но causal-интерпретация слабее, чем у ITS, "
            "DiD или Causal Impact."
        ),
        recommendation=recommendation,
        decision="scale" if p_value is not None and p_value < 0.05 and avg_delta > 0 else "hold_or_iterate",
        risk_level=risk_level,
        plots=[
            PlotSpec(title="Forecast baseline", figure=fig),
            PlotSpec(title="Delta vs baseline", figure=delta_fig),
        ],
        extra={
            "cumulative_delta": cumulative_delta,
            "delta": delta.tolist(),
            "baseline_forecast": prediction.mean.tolist(),
        },
    )
