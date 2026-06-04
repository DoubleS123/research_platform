"""Time-series anomaly detection using baseline residuals."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from research_platform.core.schemas import (
    AnalysisMethod,
    AnalysisResult,
    AnomalyDetectionParams,
    DiagnosticItem,
    PlotSpec,
    StatisticalMetric,
)
from research_platform.methods.time_series.common import (
    fit_baseline_model,
    period_mask,
    prepare_time_series,
)
from research_platform.visualization.style import BLUE, LIGHT_FILL, ORANGE, apply_plot_style


def run_anomaly_detection(df: pd.DataFrame, params: AnomalyDetectionParams) -> AnalysisResult:
    """Detect points that deviate strongly from baseline expectation."""
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
    residuals = observed - prediction.mean
    sigma = prediction.residual_std if prediction.residual_std > 0 else float(np.std(residuals, ddof=1))
    z_scores = residuals / sigma if sigma else np.zeros_like(residuals)
    anomaly_mask = np.abs(z_scores) >= params.z_threshold
    anomaly_count = int(anomaly_mask.sum())
    max_abs_z = float(np.max(np.abs(z_scores))) if len(z_scores) else 0.0
    avg_anomaly_delta = float(np.mean(residuals[anomaly_mask])) if anomaly_count else 0.0
    upper_band = prediction.mean + params.z_threshold * sigma
    lower_band = prediction.mean - params.z_threshold * sigma

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=analysis[params.date_col],
            y=observed,
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
            name="Ожидаемое значение",
            line=dict(color=ORANGE, width=2.2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(analysis[params.date_col]) + list(analysis[params.date_col][::-1]),
            y=list(upper_band) + list(lower_band[::-1]),
            fill="toself",
            fillcolor=LIGHT_FILL,
            line=dict(color="rgba(0,0,0,0)"),
            name=f"Нормальный диапазон ±{params.z_threshold:.1f}σ",
            hoverinfo="skip",
        )
    )
    if anomaly_count:
        anomalies = analysis.loc[anomaly_mask]
        fig.add_trace(
            go.Scatter(
                x=anomalies[params.date_col],
                y=observed[anomaly_mask],
                mode="markers",
                name="Аномалии",
                marker=dict(color=ORANGE, size=10, symbol="x"),
            )
        )
    fig = apply_plot_style(
        fig,
        title="Anomaly Detection: факт, baseline и аномальные точки",
        xaxis_title="Дата",
        yaxis_title=params.target_col,
    )

    z_fig = go.Figure()
    z_fig.add_trace(
        go.Bar(x=analysis[params.date_col], y=z_scores, name="z-score", marker_color=BLUE)
    )
    z_fig.add_hline(y=params.z_threshold, line_dash="dash", line_color=ORANGE)
    z_fig.add_hline(y=-params.z_threshold, line_dash="dash", line_color=ORANGE)
    z_fig = apply_plot_style(
        z_fig,
        title="Severity аномалий по z-score",
        xaxis_title="Дата",
        yaxis_title="z-score",
    )

    if anomaly_count:
        conclusion = (
            f"Найдено {anomaly_count} аномальных периодов по метрике `{params.target_col}`. "
            f"Максимальная severity: {max_abs_z:.2f}σ."
        )
        recommendation = (
            "Рекомендация: проверить даты аномалий на внешние шоки, сбои данных, маркетинговые "
            "активности или изменения продукта. Для causal-анализа эти периоды лучше объяснить "
            "или проверить отдельно."
        )
        risk_level = "medium"
    else:
        conclusion = (
            f"Аномальные периоды по метрике `{params.target_col}` не обнаружены при пороге "
            f"{params.z_threshold:.1f}σ."
        )
        recommendation = (
            "Рекомендация: использовать результат как диагностическую проверку качества ряда; "
            "отсутствие аномалий не доказывает causal-эффект."
        )
        risk_level = "low"

    return AnalysisResult(
        method=AnalysisMethod.ANOMALY_DETECTION,
        summary=conclusion,
        effect_estimate=avg_anomaly_delta if anomaly_count else None,
        assumptions=[
            "Baseline-period отражает нормальное поведение метрики.",
            "Сильные отклонения от ожидаемого диапазона являются кандидатами на аномалии.",
            "Метод диагностический: он не оценивает причинный эффект сам по себе.",
        ],
        diagnostics=[
            DiagnosticItem(
                name="baseline_observations",
                passed=len(baseline) >= 10,
                message=f"Для baseline использовано наблюдений: {len(baseline)}.",
            ),
            DiagnosticItem(
                name="anomaly_count",
                passed=anomaly_count == 0,
                message=f"Найдено аномальных точек: {anomaly_count}.",
            ),
        ],
        statistical_metrics=[
            StatisticalMetric(
                name="Anomaly count",
                value=str(anomaly_count),
                description="Количество точек за пределами ожидаемого диапазона",
            ),
            StatisticalMetric(
                name="Max severity",
                value=f"{max_abs_z:.2f}σ",
                description="Максимальное абсолютное отклонение от baseline в стандартных отклонениях",
            ),
            StatisticalMetric(
                name="Threshold",
                value=f"{params.z_threshold:.1f}σ",
                description="Порог, выше которого точка считается аномальной",
            ),
            StatisticalMetric(
                name="Avg anomaly delta",
                value=f"{avg_anomaly_delta:.4f}" if anomaly_count else "n/a",
                description="Среднее отклонение аномальных точек от baseline",
            ),
        ],
        conclusion=conclusion,
        interpretation=(
            "Anomaly Detection отвечает на вопрос, были ли в ряду необычные точки или периоды. "
            "Метод полезен перед causal-анализом: аномалии могут искажать DiD, ITS, Causal Impact "
            "и forecast baseline."
        ),
        recommendation=recommendation,
        decision="hold_or_iterate" if anomaly_count else "scale",
        risk_level=risk_level,
        plots=[
            PlotSpec(title="Anomaly detection", figure=fig),
            PlotSpec(title="Anomaly severity", figure=z_fig),
        ],
        extra={
            "anomaly_count": anomaly_count,
            "z_scores": z_scores.tolist(),
            "residuals": residuals.tolist(),
        },
    )
