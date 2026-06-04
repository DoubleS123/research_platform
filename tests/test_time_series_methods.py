from datetime import date

import pandas as pd

from research_platform.core.schemas import (
    AnomalyDetectionParams,
    ForecastBaselineParams,
    InterruptedTimeSeriesParams,
)
from research_platform.methods.anomaly_detection import run_anomaly_detection
from research_platform.methods.forecast_baseline import run_forecast_baseline
from research_platform.methods.interrupted_time_series import run_interrupted_time_series


def test_interrupted_time_series_detects_level_change() -> None:
    df = pd.read_csv("tests/fixtures/synthetic_time_series.csv", parse_dates=["date"])
    params = InterruptedTimeSeriesParams(
        date_col="date",
        target_col="target_metric",
        intervention_date=date(2022, 12, 14),
        covariate_cols=["covariate_1", "covariate_2"],
    )
    result = run_interrupted_time_series(df, params)
    assert result.effect_estimate is not None
    assert result.effect_estimate > 20
    assert result.plots


def test_forecast_baseline_detects_positive_delta() -> None:
    df = pd.read_csv("tests/fixtures/synthetic_time_series.csv", parse_dates=["date"])
    params = ForecastBaselineParams(
        date_col="date",
        target_col="target_metric",
        baseline_start=date(2012, 1, 1),
        baseline_end=date(2022, 12, 13),
        analysis_start=date(2022, 12, 14),
        analysis_end=date(2025, 9, 8),
        covariate_cols=["covariate_1", "covariate_2"],
    )
    result = run_forecast_baseline(df, params)
    assert result.effect_estimate is not None
    assert result.effect_estimate > 50
    assert len(result.plots) == 2


def test_anomaly_detection_runs() -> None:
    df = pd.read_csv("tests/fixtures/synthetic_time_series.csv", parse_dates=["date"])
    params = AnomalyDetectionParams(
        date_col="date",
        target_col="target_metric",
        baseline_start=date(2012, 1, 1),
        baseline_end=date(2022, 12, 13),
        analysis_start=date(2022, 12, 14),
        analysis_end=date(2025, 9, 8),
        covariate_cols=["covariate_1", "covariate_2"],
        z_threshold=3.0,
    )
    result = run_anomaly_detection(df, params)
    assert result.statistical_metrics
    assert result.plots
