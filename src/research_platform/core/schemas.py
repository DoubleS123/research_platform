"""Shared data models for analyses."""

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AnalysisMethod(str, Enum):
    """Supported quasi-experiment methods."""

    DID = "diff_in_diff"
    EVENT_STUDY = "event_study"
    CAUSAL_IMPACT = "causal_impact"
    ITS = "interrupted_time_series"
    FORECAST_BASELINE = "forecast_baseline"
    ANOMALY_DETECTION = "anomaly_detection"


class DatasetFormat(str, Enum):
    """Detected CSV layout."""

    PANEL = "panel"
    TIME_SERIES = "time_series"
    LONG_METRICS = "long_metrics"
    UNKNOWN = "unknown"


class ColumnProfile(BaseModel):
    """Summary of a single column."""

    name: str
    dtype: str
    n_unique: int
    n_missing: int
    sample_values: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    """Auto-detected dataset characteristics."""

    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    detected_format: DatasetFormat
    suggested_unit_col: str | None = None
    suggested_time_col: str | None = None
    suggested_group_col: str | None = None
    suggested_metric_cols: list[str] = Field(default_factory=list)
    has_treatment_start: bool = False


class MetricSpec(BaseModel):
    """Outcome metric definition."""

    name: str
    is_primary: bool = True


class DiagnosticItem(BaseModel):
    """Single diagnostic check result."""

    name: str
    passed: bool
    message: str


class PlotSpec(BaseModel):
    """Plot metadata for UI rendering."""

    title: str
    figure: Any  # plotly Figure — kept as Any to avoid pydantic import cycles


class StatisticalMetric(BaseModel):
    """Named statistical metric for UI and report rendering."""

    name: str
    value: str
    description: str = ""


class AnalysisResult(BaseModel):
    """Unified analysis output."""

    method: AnalysisMethod
    summary: str
    effect_estimate: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    relative_effect_pct: float | None = None
    assumptions: list[str] = Field(default_factory=list)
    diagnostics: list[DiagnosticItem] = Field(default_factory=list)
    statistical_metrics: list[StatisticalMetric] = Field(default_factory=list)
    conclusion: str = ""
    interpretation: str = ""
    recommendation: str = ""
    decision: str = ""
    risk_level: str = ""
    plots: list[PlotSpec] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class DidParams(BaseModel):
    """Minimal Diff-in-Diff parameters."""

    unit_col: str
    time_col: str
    group_col: str
    outcome_col: str
    treated_label: str = "treated"
    intervention_date: date | None = None
    treatment_start_col: str | None = None


class EventStudyParams(BaseModel):
    """Minimal Event Study parameters."""

    unit_col: str
    time_col: str
    outcome_col: str
    event_date_col: str | None = None
    global_event_date: date | None = None
    group_col: str | None = None
    treated_label: str = "treated"
    n_leads: int = 6
    n_lags: int = 6


class CausalImpactParams(BaseModel):
    """Minimal Causal Impact parameters."""

    date_col: str
    target_col: str
    pre_start: date
    pre_end: date
    post_start: date
    post_end: date
    covariate_cols: list[str] = Field(default_factory=list)


class InterruptedTimeSeriesParams(BaseModel):
    """Minimal Interrupted Time Series parameters."""

    date_col: str
    target_col: str
    intervention_date: date
    covariate_cols: list[str] = Field(default_factory=list)


class ForecastBaselineParams(BaseModel):
    """Minimal forecast + baseline parameters."""

    date_col: str
    target_col: str
    baseline_start: date
    baseline_end: date
    analysis_start: date
    analysis_end: date
    covariate_cols: list[str] = Field(default_factory=list)


class AnomalyDetectionParams(BaseModel):
    """Minimal anomaly detection parameters."""

    date_col: str
    target_col: str
    baseline_start: date
    baseline_end: date
    analysis_start: date
    analysis_end: date
    covariate_cols: list[str] = Field(default_factory=list)
    z_threshold: float = 3.0
