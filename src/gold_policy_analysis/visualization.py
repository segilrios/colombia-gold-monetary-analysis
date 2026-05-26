from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "demanda_energetica": "demanda",
    "precio_brent": "brent",
    "precio_cafe_centusd": "cafe",
    "bancolombia_price_usd": "bancolombia",
    "inflacion_sin_alimentos": "inflacion",
    "precio_bolsa_nacional_energetica": "bolsa_energia",
}


def plot_embedding_with_loadings(
    embedding_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot clustered embedding and top local-biplot arrows per cluster."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        embedding_df["embedding_x"],
        embedding_df["embedding_y"],
        c=embedding_df["cluster"],
        cmap="tab10",
        s=34,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
    )

    for cluster, cluster_df in embedding_df.groupby("cluster"):
        center = cluster_df[["embedding_x", "embedding_y"]].mean()
        top_loadings = (
            loadings_df[loadings_df["cluster"] == cluster]
            .sort_values("relevance", ascending=False)
            .head(3)
        )
        for _, row in top_loadings.iterrows():
            dx = row["loading_x"]
            dy = row["loading_y"]
            ax.arrow(
                center["embedding_x"],
                center["embedding_y"],
                dx,
                dy,
                width=0.006,
                head_width=0.08,
                length_includes_head=True,
                color="black",
                alpha=0.65,
            )
            ax.text(
                center["embedding_x"] + dx,
                center["embedding_y"] + dy,
                LABELS.get(row["feature"], row["feature"]),
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.65, "lw": 0},
            )

    ax.set_title("UMAP/local biplot clustering for gold-policy analysis")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
