"""Генератор демонстрационных CSV для обоих форматов данных.

Запуск: python tests/fixtures/_generate_examples.py
Все данные синтетические, seed зафиксирован для воспроизводимости.

Сценарии специально подобраны так, чтобы p-value покрывал весь спектр —
от уверенной значимости до пограничных и незначимых результатов.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
rng = np.random.default_rng(20240602)


def _save(df: pd.DataFrame, name: str) -> None:
    numeric = df.select_dtypes("number").columns
    df[numeric] = df[numeric].round(4)
    df.to_csv(OUT / name, index=False)
    print(f"{name}: {len(df)} строк, колонки: {list(df.columns)}")


def _panel(
    *,
    effect: float,
    noise_sd: float,
    n_treated: int,
    n_control: int,
    n_months: int,
    unit_prefix: str,
    intervention: str | None = None,
    waves: list[str] | None = None,
    ramp: bool = False,
    base_mean: float = 150.0,
    base_sd: float = 14.0,
    trend: float = 0.5,
    season_amp: float = 5.0,
) -> pd.DataFrame:
    """Generic panel generator with treated/control groups.

    Effect is added to treated rows on/after their treatment start. With ``waves`` the
    treatment date differs per unit (staggered); otherwise a single ``intervention`` date.
    """
    months = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    rows = []
    for i in range(n_treated + n_control):
        treated = i < n_treated
        unit = f"{unit_prefix}_{i:03d}"
        base = rng.normal(base_mean, base_sd)
        if treated:
            start = pd.Timestamp(waves[i % len(waves)]) if waves else pd.Timestamp(intervention)
        else:
            start = pd.NaT
        for m_idx, dt in enumerate(months):
            value = (
                base
                + trend * m_idx
                + season_amp * np.sin(2 * np.pi * m_idx / 12)
                + rng.normal(0, noise_sd)
            )
            if treated and dt >= start:
                if ramp:
                    months_since = (dt.to_period("M") - start.to_period("M")).n
                    value += effect * min(1.0, 0.4 + 0.3 * months_since)
                else:
                    value += effect
            rows.append(
                {
                    "unit_id": unit,
                    "date": dt.date().isoformat(),
                    "group": "treated" if treated else "control",
                    "treatment_start": start.date().isoformat() if treated else "",
                    "metric_value": value,
                }
            )
    return pd.DataFrame(rows)


def _time_series(
    *,
    n_days: int,
    launch_day: int | None,
    uplift_pct: float,
    noise_sd: float,
    with_covariate: bool,
    incidents: list[tuple[int, float]] | None = None,
    base: float = 800.0,
    trend: float = 0.4,
    season_amp: float = 22.0,
) -> pd.DataFrame:
    """Generic daily time series with optional level uplift, covariate and incidents."""
    days = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for d_idx, dt in enumerate(days):
        # stationary covariate (no random-walk drift) so it cancels cleanly in the
        # counterfactual instead of swamping the effect on long horizons.
        covariate = 1000.0 + 35 * np.sin(2 * np.pi * d_idx / 7) + rng.normal(0, 18)
        value = (
            base
            + trend * d_idx
            + season_amp * np.sin(2 * np.pi * d_idx / 7)
            + rng.normal(0, noise_sd)
        )
        if with_covariate:
            value += 0.4 * (covariate - 1000)
        if launch_day is not None and d_idx >= launch_day:
            value *= 1.0 + uplift_pct
        if incidents:
            for inc_day, factor in incidents:
                if d_idx == inc_day:
                    value *= factor
        row = {"date": dt.date().isoformat(), "target_metric": value}
        if with_covariate:
            row["covariate_1"] = covariate
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PANEL
# ---------------------------------------------------------------------------
def panel_staggered_rollout() -> pd.DataFrame:
    """Раскатка по городам волнами. Эффект есть, значимость умеренная."""
    return _panel(
        effect=6.0,
        noise_sd=9.0,
        n_treated=14,
        n_control=10,
        n_months=16,
        unit_prefix="city",
        waves=["2024-07-01", "2024-10-01", "2025-01-01"],
        ramp=True,
        base_mean=120.0,
    )


def panel_pricing_negative() -> pd.DataFrame:
    """Повышение цены снизило заказы. Значимый, но не «лабораторный» отрицательный эффект."""
    return _panel(
        effect=-6.0,
        noise_sd=13.0,
        n_treated=12,
        n_control=12,
        n_months=12,
        unit_prefix="store",
        intervention="2024-08-01",
        base_mean=200.0,
        season_amp=6.0,
    )


def panel_null_effect() -> pd.DataFrame:
    """Фича раскатана, но реального эффекта почти нет — результат НЕ значим."""
    return _panel(
        effect=1.0,
        noise_sd=12.0,
        n_treated=12,
        n_control=12,
        n_months=14,
        unit_prefix="region",
        intervention="2024-08-01",
        base_mean=90.0,
    )


# ---------------------------------------------------------------------------
# TIME SERIES
# ---------------------------------------------------------------------------
def ts_campaign_uplift() -> pd.DataFrame:
    """Маркетинговая кампания дала заметный, но не «лабораторный» uplift (умеренная значимость)."""
    return _time_series(
        n_days=180,
        launch_day=140,
        uplift_pct=0.025,
        noise_sd=45.0,
        with_covariate=True,
    )


def ts_marginal_effect() -> pd.DataFrame:
    """Слабый uplift на коротком post-периоде — p около порога 0.05 (пограничный)."""
    return _time_series(
        n_days=120,
        launch_day=95,
        uplift_pct=0.011,
        noise_sd=44.0,
        with_covariate=False,
    )


def ts_anomaly_incidents() -> pd.DataFrame:
    """Ряд с инцидентами разной силы: явный сбой, всплеск и слабая аномалия у порога.

    Подобрано так, чтобы при пороге 3σ ловились сильные инциденты, а слабый блип
    отсекался — и появлялся только при снижении порога до 2.5σ (демонстрация чувствительности).
    """
    return _time_series(
        n_days=150,
        launch_day=None,
        uplift_pct=0.0,
        noise_sd=10.0,
        with_covariate=False,
        incidents=[(95, 0.93), (96, 0.93), (120, 1.045), (140, 1.03)],
    )


if __name__ == "__main__":
    _save(panel_staggered_rollout(), "panel_staggered_rollout.csv")
    _save(panel_pricing_negative(), "panel_pricing_negative.csv")
    _save(panel_null_effect(), "panel_null_effect.csv")
    _save(ts_campaign_uplift(), "ts_campaign_uplift.csv")
    _save(ts_marginal_effect(), "ts_marginal_effect.csv")
    _save(ts_anomaly_incidents(), "ts_anomaly_incidents.csv")
