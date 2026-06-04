from datetime import date

import pandas as pd

from research_platform.core.schemas import CausalImpactParams
from research_platform.methods.causal_impact import run_causal_impact


def test_causal_impact_detects_lift() -> None:
    df = pd.read_csv("tests/fixtures/synthetic_time_series.csv", parse_dates=["date"])
    params = CausalImpactParams(
        date_col="date",
        target_col="target_metric",
        pre_start=date(2012, 1, 1),
        pre_end=date(2022, 12, 13),
        post_start=date(2022, 12, 14),
        post_end=date(2025, 9, 8),
        covariate_cols=["covariate_1", "covariate_2"],
    )
    result = run_causal_impact(df, params)
    assert result.effect_estimate is not None
    assert result.effect_estimate > 50
    assert all(plot.figure.data for plot in result.plots)
