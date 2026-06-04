import pandas as pd

from research_platform.core.schemas import DidParams
from research_platform.methods.did import run_did


def test_did_positive_effect() -> None:
    df = pd.read_csv("tests/fixtures/synthetic_panel.csv", parse_dates=["date", "treatment_start"])
    params = DidParams(
        unit_col="unit_id",
        time_col="date",
        group_col="group",
        outcome_col="metric_value",
        treated_label="treated",
        treatment_start_col="treatment_start",
    )
    result = run_did(df, params)
    assert result.effect_estimate is not None
    assert result.effect_estimate > 5
