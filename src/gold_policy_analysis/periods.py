from __future__ import annotations

import pandas as pd


def assign_executive_periods(
    df: pd.DataFrame,
    date_column: str,
    periods: list[dict[str, str]],
    output_column: str = "executive_period",
) -> pd.DataFrame:
    """Assign each observation to a Colombian executive period."""
    out = df.copy()
    out[output_column] = "outside_configured_periods"
    dates = pd.to_datetime(out[date_column])

    for period in periods:
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        mask = (dates >= start) & (dates <= end)
        out.loc[mask, output_column] = period["name"]

    return out


def classify_policy_stance(
    df: pd.DataFrame,
    rate_column: str,
    output_column: str = "policy_stance",
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Classify monetary policy stance from month-to-month rate movements."""
    out = df.copy()
    change = out[rate_column].diff()
    out["policy_rate_change"] = change
    out[output_column] = "neutral"
    out.loc[change > tolerance, output_column] = "tightening"
    out.loc[change < -tolerance, output_column] = "easing"
    return out
