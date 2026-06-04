from datetime import date

import pandas as pd

from research_platform.core.schemas import EventStudyParams
from research_platform.methods.event_study import run_event_study


def test_event_study_runs() -> None:
    df = pd.read_csv("tests/fixtures/synthetic_panel.csv", parse_dates=["date", "treatment_start"])
    params = EventStudyParams(
        unit_col="unit_id",
        time_col="date",
        outcome_col="metric_value",
        group_col="group",
        treated_label="treated",
        global_event_date=date(2025, 1, 1),
        n_leads=2,
        n_lags=2,
    )
    result = run_event_study(df, params)
    assert result.plots
    assert "periods" in result.extra


def test_event_study_auto_selects_available_baseline() -> None:
    rows = []
    for unit_idx in range(8):
        for dt, value in [
            ("2025-01-01", 100 + unit_idx),
            ("2025-03-01", 115 + unit_idx),
            ("2025-04-01", 118 + unit_idx),
        ]:
            rows.append(
                {
                    "unit_id": f"u_{unit_idx}",
                    "date": dt,
                    "group": "treated",
                    "treatment_start": "2025-03-01",
                    "metric_value": value,
                }
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["treatment_start"] = pd.to_datetime(df["treatment_start"])
    params = EventStudyParams(
        unit_col="unit_id",
        time_col="date",
        outcome_col="metric_value",
        event_date_col="treatment_start",
        group_col="group",
        treated_label="treated",
        n_leads=3,
        n_lags=2,
    )
    result = run_event_study(df, params)
    assert result.extra["baseline_period"] == -2
    assert result.plots
