"""Shared time-series preparation and baseline modeling."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm

from research_platform.utils.errors import EstimationError, ValidationError


@dataclass(frozen=True)
class BaselinePrediction:
    """Prediction output for forecast-style methods."""

    model: object
    prediction: object
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    residual_std: float


def prepare_time_series(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    covariate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Parse, aggregate and sort a time-series dataframe.

    If there are multiple rows per date, numeric target/covariates are averaged by date.
    """
    covariates = covariate_cols or []
    required = [date_col, target_col, *covariates]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValidationError(f"Time series: missing columns {missing}")

    work = df[required].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, target_col])
    if work.empty:
        raise ValidationError("Time series: no valid rows after date/target parsing.")

    for col in [target_col, *covariates]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[target_col])

    aggregation = {col: "mean" for col in [target_col, *covariates]}
    work = work.groupby(date_col, as_index=False).agg(aggregation).sort_values(date_col)
    work["trend"] = np.arange(len(work), dtype=float)
    return work


def period_mask(series: pd.Series, start: date, end: date) -> pd.Series:
    """Return mask for inclusive date interval."""
    return (series >= pd.Timestamp(start)) & (series <= pd.Timestamp(end))


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    covariate_cols: list[str] | None = None,
    include_const: bool = True,
) -> pd.DataFrame:
    """Build common trend/covariate feature matrix."""
    covariates = covariate_cols or []
    features = pd.DataFrame({"trend": df["trend"].astype(float)}, index=df.index)
    for col in covariates:
        median = float(pd.to_numeric(df[col], errors="coerce").median())
        features[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)
    if include_const:
        return sm.add_constant(features, has_constant="add")
    return features


def fit_baseline_model(
    *,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    target_col: str,
    covariate_cols: list[str] | None = None,
) -> BaselinePrediction:
    """Fit OLS trend baseline and predict on another period."""
    if len(train_df) < 10:
        raise EstimationError("Baseline model requires at least 10 baseline observations.")

    x_train = build_feature_matrix(train_df, covariate_cols=covariate_cols)
    x_predict = build_feature_matrix(predict_df, covariate_cols=covariate_cols)
    x_predict = x_predict.reindex(columns=x_train.columns, fill_value=1.0)
    y_train = train_df[target_col].astype(float)

    try:
        model = sm.OLS(y_train, x_train).fit()
        prediction = model.get_prediction(x_predict)
        mean = np.asarray(prediction.predicted_mean, dtype=float)
        frame = prediction.summary_frame(alpha=0.05)
        lower = frame["mean_ci_lower"].to_numpy(dtype=float)
        upper = frame["mean_ci_upper"].to_numpy(dtype=float)
        residual_std = float(np.std(model.resid, ddof=1))
    except Exception as exc:
        raise EstimationError(f"Baseline model failed: {exc}") from exc

    if len(mean) == 0 or np.isnan(mean).all():
        raise EstimationError("Baseline model produced empty forecast.")

    return BaselinePrediction(
        model=model,
        prediction=prediction,
        mean=mean,
        lower=lower,
        upper=upper,
        residual_std=residual_std,
    )


def relative_effect_pct(effect: float, baseline: float) -> float | None:
    """Calculate relative effect safely."""
    if baseline == 0 or np.isnan(baseline):
        return None
    return 100.0 * effect / baseline
