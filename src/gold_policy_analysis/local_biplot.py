from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def _affine_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    augmented = np.column_stack([source, np.ones(source.shape[0])])
    coefficients, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    return coefficients


def _affine_apply(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    augmented = np.column_stack([points, np.ones(points.shape[0])])
    return augmented @ coefficients


def compute_local_biplot(
    scaled_features: np.ndarray,
    embedding: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Estimate local PCA loadings and align them to the embedding space."""
    rows: list[dict[str, float | int | str]] = []
    unique_labels = sorted(np.unique(labels))

    for cluster in unique_labels:
        mask = labels == cluster
        n_obs = int(mask.sum())
        if n_obs < 3:
            continue

        x_local = scaled_features[mask]
        z_local = embedding[mask]
        pca = PCA(n_components=2, random_state=42)
        local_scores = pca.fit_transform(x_local)
        local_scores = MinMaxScaler(feature_range=(-1, 1)).fit_transform(local_scores)

        coefficients = _affine_fit(local_scores, z_local)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loading_points = _affine_apply(loadings, coefficients)
        cluster_center = z_local.mean(axis=0)
        vectors = loading_points - cluster_center
        relevance = np.abs(loadings).sum(axis=1)
        relevance = relevance / relevance.sum() if relevance.sum() else relevance

        for feature, vector, rel in zip(feature_names, vectors, relevance, strict=False):
            rows.append(
                {
                    "cluster": int(cluster),
                    "feature": feature,
                    "loading_x": float(vector[0]),
                    "loading_y": float(vector[1]),
                    "relevance": float(rel),
                    "n_obs": n_obs,
                }
            )

    return pd.DataFrame(rows)
