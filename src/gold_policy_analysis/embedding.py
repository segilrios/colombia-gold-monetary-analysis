from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class EmbeddingResult:
    coordinates: np.ndarray
    scaled_features: np.ndarray
    method: str
    feature_names: list[str]


def standardize_features(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Scale features for distance-based embedding."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, list(features.columns)


def compute_embedding(
    features: pd.DataFrame,
    n_neighbors: int = 12,
    min_dist: float = 0.15,
    metric: str = "euclidean",
    random_state: int = 42,
) -> EmbeddingResult:
    """Compute a 2D UMAP embedding, with PCA fallback when UMAP is unavailable."""
    scaled, feature_names = standardize_features(features)
    n_neighbors = max(2, min(int(n_neighbors), scaled.shape[0] - 1))

    try:
        from umap import UMAP

        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        coordinates = reducer.fit_transform(scaled)
        method = "umap"
    except ImportError:
        reducer = PCA(n_components=2, random_state=random_state)
        coordinates = reducer.fit_transform(scaled)
        method = "pca_fallback"

    return EmbeddingResult(
        coordinates=coordinates,
        scaled_features=scaled,
        method=method,
        feature_names=feature_names,
    )


def cluster_embedding(
    coordinates: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster the 2D embedding with KMeans."""
    n_clusters = max(2, min(int(n_clusters), coordinates.shape[0]))
    model = KMeans(n_clusters=n_clusters, n_init=25, random_state=random_state)
    return model.fit_predict(coordinates)
