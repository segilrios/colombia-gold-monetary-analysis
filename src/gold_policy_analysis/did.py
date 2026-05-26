from __future__ import annotations

import numpy as np
import pandas as pd


def build_policy_event_design(
    df: pd.DataFrame,
    date_column: str,
    outcome_column: str,
    policy_column: str,
    cluster_column: str = "cluster",
    period_column: str = "executive_period",
    event_window: int = 6,
    policy_change_quantile: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a DiD-ready event table and descriptive pre/post contrasts.

    The current data is a national time series, so this is not yet a causal DiD
    estimator. It produces the design columns needed to move toward DiD once a
    control unit, untreated comparison group or richer panel is defined.
    """
    out = df.sort_values(date_column).reset_index(drop=True).copy()
    out["policy_change_abs"] = out[policy_column].diff().abs()
    positive_changes = out.loc[out["policy_change_abs"] > 0, "policy_change_abs"].dropna()
    threshold = positive_changes.quantile(policy_change_quantile)
    if pd.isna(threshold):
        threshold = np.inf

    out["policy_event"] = out["policy_change_abs"] >= threshold
    out.loc[out["policy_change_abs"].isna(), "policy_event"] = False
    event_positions = out.index[out["policy_event"]].to_numpy()

    out["event_id"] = pd.NA
    out["event_time"] = np.nan
    out["post_policy_event"] = False
    out["treated_policy_window"] = False

    for event_number, event_position in enumerate(event_positions, start=1):
        lo = max(0, event_position - event_window)
        hi = min(len(out) - 1, event_position + event_window)
        idx = np.arange(lo, hi + 1)
        empty = out.loc[idx, "event_id"].isna()
        idx = idx[empty.to_numpy()]
        out.loc[idx, "event_id"] = event_number
        out.loc[idx, "event_time"] = idx - event_position
        out.loc[idx, "post_policy_event"] = out.loc[idx, "event_time"] >= 0
        out.loc[idx, "treated_policy_window"] = True

    out["did_post_x_treated"] = (
        out["post_policy_event"].astype(int) * out["treated_policy_window"].astype(int)
    )

    contrasts = []
    for event_number in sorted(out["event_id"].dropna().unique()):
        event_df = out[out["event_id"] == event_number]
        pre = event_df[event_df["event_time"] < 0][outcome_column]
        post = event_df[event_df["event_time"] >= 0][outcome_column]
        event_row = event_df.loc[event_df["event_time"].abs().idxmin()]
        contrasts.append(
            {
                "event_id": int(event_number),
                "event_date": event_row[date_column],
                "policy_column": policy_column,
                "policy_change_abs": float(event_row["policy_change_abs"]),
                "executive_period": event_row.get(period_column),
                "cluster": event_row.get(cluster_column),
                "pre_mean": float(pre.mean()) if len(pre) else np.nan,
                "post_mean": float(post.mean()) if len(post) else np.nan,
                "post_minus_pre": float(post.mean() - pre.mean()) if len(pre) and len(post) else np.nan,
                "n_pre": int(len(pre)),
                "n_post": int(len(post)),
            }
        )

    return out, pd.DataFrame(contrasts)
