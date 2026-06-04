"""Input validation helpers."""

from datetime import date

import pandas as pd

from research_platform.core.schemas import CausalImpactParams, DidParams, EventStudyParams
from research_platform.utils.errors import ValidationError


def _require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValidationError(f"{context}: missing columns {missing}")


def _parse_dates(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception as exc:
        raise ValidationError(f"Cannot parse dates: {exc}") from exc


def validate_did_inputs(df: pd.DataFrame, params: DidParams) -> pd.DataFrame:
    """Validate and prepare data for Diff-in-Diff."""
    _require_columns(df, [params.unit_col, params.time_col, params.group_col, params.outcome_col], "DiD")
    work = df.copy()
    work[params.time_col] = _parse_dates(work[params.time_col])
    if work[params.time_col].isna().any():
        raise ValidationError("DiD: invalid dates in time column.")
    if params.treatment_start_col:
        work[params.treatment_start_col] = _parse_dates(work[params.treatment_start_col])
    if params.intervention_date is None and not params.treatment_start_col:
        raise ValidationError("DiD: provide intervention_date or treatment_start_col.")

    group_values = work[params.group_col].dropna().astype(str)
    labels_lower = group_values.str.lower()
    treated_lower = params.treated_label.lower()
    if treated_lower not in set(labels_lower):
        available = sorted(group_values.unique())
        raise ValidationError(
            f"DiD: treated_label '{params.treated_label}' not found in column "
            f"'{params.group_col}'. Available values: {available}."
        )
    if (labels_lower != treated_lower).sum() == 0:
        raise ValidationError(
            f"DiD: no control rows found — every value in '{params.group_col}' equals the "
            "treated label. Provide a control group."
        )
    return work


def validate_event_study_inputs(df: pd.DataFrame, params: EventStudyParams) -> pd.DataFrame:
    """Validate and prepare data for Event Study."""
    cols = [params.unit_col, params.time_col, params.outcome_col]
    if params.group_col:
        cols.append(params.group_col)
    if params.event_date_col:
        cols.append(params.event_date_col)
    _require_columns(df, cols, "Event Study")
    if params.global_event_date is None and not params.event_date_col:
        raise ValidationError("Event Study: provide global_event_date or event_date_col.")
    work = df.copy()
    work[params.time_col] = _parse_dates(work[params.time_col])
    if params.event_date_col:
        work[params.event_date_col] = _parse_dates(work[params.event_date_col])
    return work


def validate_causal_impact_inputs(df: pd.DataFrame, params: CausalImpactParams) -> pd.DataFrame:
    """Validate and prepare data for Causal Impact."""
    _require_columns(df, [params.date_col, params.target_col], "Causal Impact")
    work = df.copy()
    work[params.date_col] = _parse_dates(work[params.date_col])
    if work[params.date_col].isna().any():
        raise ValidationError("Causal Impact: invalid dates.")
    for col in params.covariate_cols:
        if col not in work.columns:
            raise ValidationError(f"Causal Impact: covariate column '{col}' not found.")
    if params.pre_end >= params.post_start:
        raise ValidationError("Causal Impact: pre period must end before post period starts.")
    return work


def ensure_period_coverage(dates: pd.Series, start: date, end: date, label: str) -> None:
    """Check that the dataset has observations in [start, end]."""
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if not mask.any():
        raise ValidationError(f"{label}: no observations between {start} and {end}.")
