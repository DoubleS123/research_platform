import pandas as pd

from research_platform.core.profiling import profile_dataset
from research_platform.core.recommendation import recommend_method
from research_platform.core.schemas import AnalysisMethod, DatasetFormat


def test_recommend_panel_for_did() -> None:
    df = pd.read_csv("tests/fixtures/synthetic_panel.csv")
    profile = profile_dataset(df)
    assert profile.detected_format == DatasetFormat.PANEL
    rec = recommend_method(profile)
    assert rec.method == AnalysisMethod.DID


def test_recommend_timeseries_for_causal_impact() -> None:
    df = pd.read_csv("tests/fixtures/synthetic_time_series.csv", parse_dates=["date"])
    profile = profile_dataset(df)
    rec = recommend_method(profile)
    assert rec.method == AnalysisMethod.CAUSAL_IMPACT
