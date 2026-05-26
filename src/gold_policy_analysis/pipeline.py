from __future__ import annotations

from pathlib import Path

import pandas as pd

from gold_policy_analysis.config import load_config
from gold_policy_analysis.data import feature_matrix, load_raw_excel, prepare_time_series
from gold_policy_analysis.did import build_policy_event_design
from gold_policy_analysis.embedding import cluster_embedding, compute_embedding
from gold_policy_analysis.local_biplot import compute_local_biplot
from gold_policy_analysis.periods import assign_executive_periods, classify_policy_stance
from gold_policy_analysis.visualization import plot_embedding_with_loadings


def _cluster_profiles(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> pd.DataFrame:
    profile_columns = [target_column, *feature_columns]
    profiles = df.groupby("cluster")[profile_columns].agg(["count", "mean", "std"]).round(4)
    profiles.columns = ["_".join(column).strip("_") for column in profiles.columns]

    period_mix = pd.crosstab(df["cluster"], df["executive_period"], normalize="index").round(4)
    period_mix = period_mix.add_prefix("period_share_")
    stance_mix = pd.crosstab(df["cluster"], df["policy_stance"], normalize="index").round(4)
    stance_mix = stance_mix.add_prefix("stance_share_")
    return profiles.join(period_mix, how="left").join(stance_mix, how="left").reset_index()


def run_policy_map(config_path: str | Path = "configs/project.yaml") -> dict[str, Path | str]:
    """Run the first UMAP/local-biplot/clustering/DiD scaffold."""
    config = load_config(config_path)
    data_config = config["data"]
    analysis_config = config["analysis"]
    policy_config = config["policy"]

    date_column = data_config["date_column"]
    target_column = data_config["target_column"]
    feature_columns = data_config["feature_columns"]
    rate_column = policy_config["rate_column"]

    raw = load_raw_excel(data_config["raw_path"], sheet_name=data_config.get("sheet_name", 0))
    prepared = prepare_time_series(raw, date_column, target_column, feature_columns)
    prepared = assign_executive_periods(
        prepared,
        date_column=date_column,
        periods=policy_config["executive_periods"],
    )
    prepared = classify_policy_stance(prepared, rate_column=rate_column)

    features = feature_matrix(prepared, feature_columns)
    model_df = prepared.loc[features.index].reset_index(drop=True)
    features = features.reset_index(drop=True)

    umap_config = analysis_config.get("umap", {})
    embedding_result = compute_embedding(
        features,
        n_neighbors=umap_config.get("n_neighbors", 12),
        min_dist=umap_config.get("min_dist", 0.15),
        metric=umap_config.get("metric", "euclidean"),
        random_state=analysis_config.get("random_state", 42),
    )
    labels = cluster_embedding(
        embedding_result.coordinates,
        n_clusters=analysis_config.get("n_clusters", 4),
        random_state=analysis_config.get("random_state", 42),
    )

    model_df["embedding_x"] = embedding_result.coordinates[:, 0]
    model_df["embedding_y"] = embedding_result.coordinates[:, 1]
    model_df["cluster"] = labels
    model_df["embedding_method"] = embedding_result.method

    loadings = compute_local_biplot(
        embedding_result.scaled_features,
        embedding_result.coordinates,
        labels,
        embedding_result.feature_names,
    )
    profiles = _cluster_profiles(model_df, feature_columns, target_column)

    did_config = analysis_config.get("did", {})
    did_design, did_contrasts = build_policy_event_design(
        model_df,
        date_column=date_column,
        outcome_column=f"{target_column}_return",
        policy_column=rate_column,
        event_window=did_config.get("event_window", 6),
        policy_change_quantile=did_config.get("policy_change_quantile", 0.75),
    )

    processed_dir = Path("data/processed")
    tables_dir = Path("reports/tables")
    figures_dir = Path("reports/figures")
    for directory in (processed_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    embedding_path = processed_dir / "policy_umap_embedding.csv"
    loadings_path = tables_dir / "local_biplot_loadings.csv"
    profiles_path = tables_dir / "cluster_profiles.csv"
    did_design_path = processed_dir / "policy_event_did_design.csv"
    did_contrasts_path = tables_dir / "policy_event_contrasts.csv"
    figure_path = figures_dir / "policy_umap_local_biplot_clusters.png"

    model_df.to_csv(embedding_path, index=False)
    loadings.to_csv(loadings_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    did_design.to_csv(did_design_path, index=False)
    did_contrasts.to_csv(did_contrasts_path, index=False)
    plot_embedding_with_loadings(model_df, loadings, figure_path)

    return {
        "embedding_method": embedding_result.method,
        "embedding": embedding_path,
        "local_biplot_loadings": loadings_path,
        "cluster_profiles": profiles_path,
        "did_design": did_design_path,
        "did_contrasts": did_contrasts_path,
        "figure": figure_path,
    }
