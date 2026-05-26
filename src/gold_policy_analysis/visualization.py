from __future__ import annotations

from pathlib import Path
from math import ceil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import pandas as pd


LABELS = {
    "demanda_energetica": "demanda",
    "precio_brent": "brent",
    "precio_cafe_centusd": "cafe",
    "bancolombia_price_usd": "bancolombia",
    "inflacion_sin_alimentos": "inflacion",
    "precio_bolsa_nacional_energetica": "bolsa_energia",
}

PERIOD_LABELS = {
    "pastrana": "Pastrana",
    "uribe_i": "Uribe I",
    "uribe_ii": "Uribe II",
    "santos_i": "Santos I",
    "santos_ii": "Santos II",
    "duque": "Duque",
    "petro": "Petro",
}

CLUSTER_COLORS = ["#1f77b4", "#d62728", "#e377c2", "#17becf", "#9467bd", "#ff7f0e"]


def _cluster_style(max_cluster: int) -> tuple[ListedColormap, BoundaryNorm, list[str]]:
    colors = [CLUSTER_COLORS[idx % len(CLUSTER_COLORS)] for idx in range(max_cluster + 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([idx - 0.5 for idx in range(max_cluster + 2)], cmap.N)
    return cmap, norm, colors


def plot_embedding_with_loadings(
    embedding_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot clustered embedding and top local-biplot arrows per cluster."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_cluster = int(embedding_df["cluster"].max())
    cmap, norm, _ = _cluster_style(max_cluster)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        embedding_df["embedding_x"],
        embedding_df["embedding_y"],
        c=embedding_df["cluster"],
        cmap=cmap,
        norm=norm,
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


def plot_embedding_by_executive_period(
    embedding_df: pd.DataFrame,
    output_path: str | Path,
    period_column: str = "executive_period",
) -> None:
    """Facet the embedding so each panel highlights one executive period."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    periods = embedding_df.drop_duplicates(period_column)[period_column].tolist()
    ncols = 4
    nrows = ceil(len(periods) / ncols)
    xlim = (
        embedding_df["embedding_x"].min() - 1,
        embedding_df["embedding_x"].max() + 1,
    )
    ylim = (
        embedding_df["embedding_y"].min() - 1,
        embedding_df["embedding_y"].max() + 1,
    )
    max_cluster = int(embedding_df["cluster"].max())
    cmap, norm, colors = _cluster_style(max_cluster)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8), sharex=True, sharey=True)
    axes_list = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, period in zip(axes_list, periods, strict=False):
        period_df = embedding_df[embedding_df[period_column] == period]
        ax.scatter(
            embedding_df["embedding_x"],
            embedding_df["embedding_y"],
            color="#d5d5d5",
            s=8,
            alpha=0.25,
            linewidth=0,
        )
        ax.scatter(
            period_df["embedding_x"],
            period_df["embedding_y"],
            c=period_df["cluster"],
            cmap=cmap,
            norm=norm,
            s=20,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.25,
        )
        title = PERIOD_LABELS.get(period, period.replace("_", " ").title())
        ax.set_title(f"{title} ({len(period_df):,})", fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.14)

    for ax in axes_list[len(periods) :]:
        ax.axis("off")

    for ax in axes_list[-ncols:]:
        ax.set_xlabel("Embedding 1")
    for ax in axes_list[::ncols]:
        ax.set_ylabel("Embedding 2")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=str(cluster),
            markerfacecolor=colors[cluster],
            markersize=7,
        )
        for cluster in range(max_cluster + 1)
    ]
    fig.legend(handles=handles, title="Cluster", loc="lower center", ncols=max_cluster + 1)
    fig.suptitle("UMAP embedding highlighted by Colombian executive period", fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
