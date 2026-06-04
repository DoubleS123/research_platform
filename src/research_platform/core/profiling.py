"""Dataset profiling and column suggestions."""

import pandas as pd

from research_platform.core.schemas import ColumnProfile, DatasetFormat, DatasetProfile

_UNIT_HINTS = {"unit_id", "unit", "entity_id", "store_id", "user_id", "region_id", "account_id"}
_TIME_HINTS = {"date", "dt", "day", "period", "week", "month", "timestamp"}
_GROUP_HINTS = {"group", "treatment_group", "cohort", "arm", "is_treated"}
_TREATMENT_START_HINTS = {"treatment_start", "event_date", "intervention_date"}
_METRIC_HINTS = {"metric_value", "target_metric", "revenue", "orders", "value", "outcome"}


def _match_column(columns: list[str], hints: set[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for hint in hints:
        if hint in lower_map:
            return lower_map[hint]
    return None


def _detect_format(columns: list[str]) -> DatasetFormat:
    lower = {c.lower() for c in columns}
    if "metric_name" in lower and "metric_value" in lower:
        return DatasetFormat.LONG_METRICS
    if _match_column(columns, _UNIT_HINTS) and _match_column(columns, _GROUP_HINTS):
        return DatasetFormat.PANEL
    if _match_column(columns, _TIME_HINTS) and (
        _match_column(columns, {"target_metric"}) or any("metric" in c.lower() for c in columns)
    ):
        return DatasetFormat.TIME_SERIES
    return DatasetFormat.UNKNOWN


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """
    Build a dataset profile with column stats and mapping suggestions.

    Args:
        df: Input dataframe.

    Returns:
        DatasetProfile with detected format and suggested columns.
    """
    columns = list(df.columns)
    col_profiles: list[ColumnProfile] = []
    for col in columns:
        series = df[col]
        sample = series.dropna().astype(str).head(3).tolist()
        col_profiles.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                n_unique=int(series.nunique(dropna=True)),
                n_missing=int(series.isna().sum()),
                sample_values=sample,
            )
        )

    numeric_cols = [
        c
        for c in columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in {_match_column(columns, _UNIT_HINTS), _match_column(columns, _TIME_HINTS)}
    ]
    suggested_metric = _match_column(columns, _METRIC_HINTS)
    metric_cols = [suggested_metric] if suggested_metric else numeric_cols[:3]

    return DatasetProfile(
        n_rows=len(df),
        n_cols=len(columns),
        columns=col_profiles,
        detected_format=_detect_format(columns),
        suggested_unit_col=_match_column(columns, _UNIT_HINTS),
        suggested_time_col=_match_column(columns, _TIME_HINTS),
        suggested_group_col=_match_column(columns, _GROUP_HINTS),
        suggested_metric_cols=[m for m in metric_cols if m],
        has_treatment_start=_match_column(columns, _TREATMENT_START_HINTS) is not None,
    )
