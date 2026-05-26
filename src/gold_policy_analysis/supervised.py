from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from gold_policy_analysis.config import load_config
from gold_policy_analysis.data import load_raw_excel, resolve_raw_path
from gold_policy_analysis.periods import assign_executive_periods, classify_policy_stance


@dataclass(frozen=True)
class SupervisedDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    theory_feature_columns: list[str]
    horizon_columns: dict[int, str]


class ZeroReturnRegressor(BaseEstimator, RegressorMixin):
    """Naive baseline for log-return forecasting."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ZeroReturnRegressor":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class GreyBoxRandomForest(BaseEstimator, RegressorMixin):
    """Linear macro block plus Random Forest on residuals."""

    def __init__(
        self,
        theory_features: list[str],
        alpha_grid: list[float] | None = None,
        random_state: int = 42,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 3,
    ) -> None:
        self.theory_features = theory_features
        self.alpha_grid = alpha_grid or [0.01, 0.1, 1.0, 10.0, 100.0]
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GreyBoxRandomForest":
        self.feature_names_in_ = list(X.columns)
        available_theory = [column for column in self.theory_features if column in X.columns]
        if not available_theory:
            raise ValueError("No theory features are available for the grey-box linear block.")
        self.available_theory_features_ = available_theory

        self.linear_block_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.array(self.alpha_grid, dtype=float))),
            ]
        )
        self.linear_block_.fit(X[available_theory], y)
        residuals = y.to_numpy() - self.linear_block_.predict(X[available_theory])

        self.residual_forest_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.residual_forest_.fit(X, residuals)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        linear_prediction = self.linear_block_.predict(X[self.available_theory_features_])
        residual_prediction = self.residual_forest_.predict(X[self.feature_names_in_])
        return linear_prediction + residual_prediction

    def predict_components(self, X: pd.DataFrame) -> pd.DataFrame:
        linear_prediction = self.linear_block_.predict(X[self.available_theory_features_])
        residual_prediction = self.residual_forest_.predict(X[self.feature_names_in_])
        return pd.DataFrame(
            {
                "linear_macro_component": linear_prediction,
                "rf_residual_component": residual_prediction,
                "grey_box_prediction": linear_prediction + residual_prediction,
            },
            index=X.index,
        )


def _resample_frame(df: pd.DataFrame, date_column: str, frequency: str) -> pd.DataFrame:
    frequency = frequency.lower()
    if frequency in {"daily", "d"}:
        return df.sort_values(date_column).reset_index(drop=True)

    if frequency in {"monthly", "m", "me"}:
        rule = "ME"
    elif frequency in {"weekly", "w"}:
        rule = "W"
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

    numeric = df.set_index(date_column).sort_index()
    resampled = numeric.resample(rule).last().dropna(how="all")
    return resampled.reset_index()


def load_supervised_base(config: dict[str, Any]) -> pd.DataFrame:
    """Load, normalize, resample and annotate the macro-gold dataset."""
    data_config = config["data"]
    supervised_config = config.get("supervised", {})
    policy_config = config["policy"]

    date_column = data_config["date_column"]
    target_column = data_config["target_column"]
    feature_columns = data_config["feature_columns"]
    required = [date_column, target_column, *feature_columns]

    raw_path = resolve_raw_path(data_config["raw_path"])
    raw = load_raw_excel(raw_path, sheet_name=data_config.get("sheet_name", 0))
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    base = raw[required].copy()
    base[date_column] = pd.to_datetime(base[date_column], errors="coerce")
    for column in [target_column, *feature_columns]:
        base[column] = pd.to_numeric(base[column], errors="coerce")
    base = base.dropna(subset=[date_column, target_column]).sort_values(date_column)

    frequency = supervised_config.get("frequency", "monthly")
    base = _resample_frame(base, date_column, frequency)
    base = base.dropna(subset=[target_column]).reset_index(drop=True)
    base = assign_executive_periods(
        base,
        date_column=date_column,
        periods=policy_config["executive_periods"],
    )
    base = classify_policy_stance(base, rate_column=policy_config["rate_column"])
    base[f"{target_column}_log_return_1"] = np.log(base[target_column] / base[target_column].shift(1))
    return base.replace([np.inf, -np.inf], np.nan)


def _monthly_mode(series: pd.Series) -> float | int | str | None:
    mode = series.dropna().mode()
    if mode.empty:
        return None
    return mode.iloc[0]


def add_umap_context(
    base: pd.DataFrame,
    date_column: str,
    umap_context_path: str | Path,
) -> pd.DataFrame:
    """Join previous UMAP cluster context when the generated file exists."""
    path = Path(umap_context_path)
    if not path.exists():
        return base

    context = pd.read_csv(path)
    if date_column not in context.columns or "cluster" not in context.columns:
        return base

    context[date_column] = pd.to_datetime(context[date_column], errors="coerce")
    context = context.dropna(subset=[date_column])
    aggregations: dict[str, str | Any] = {"cluster": _monthly_mode}
    if "embedding_x" in context.columns:
        aggregations["embedding_x"] = "mean"
    if "embedding_y" in context.columns:
        aggregations["embedding_y"] = "mean"

    monthly = context.set_index(date_column).resample("ME").agg(aggregations).reset_index()
    monthly = monthly.rename(
        columns={
            "cluster": "umap_cluster",
            "embedding_x": "umap_embedding_x",
            "embedding_y": "umap_embedding_y",
        }
    )
    monthly["umap_cluster"] = monthly["umap_cluster"].astype("Int64").astype(str)
    return base.merge(monthly, on=date_column, how="left")


def build_supervised_dataset_from_config(config: dict[str, Any]) -> SupervisedDataset:
    """Build the time-series supervised learning matrix from a loaded config."""
    data_config = config["data"]
    supervised_config = config.get("supervised", {})

    date_column = data_config["date_column"]
    target_column = data_config["target_column"]
    feature_columns = data_config["feature_columns"]
    horizons = [int(value) for value in supervised_config.get("horizons", [1, 3, 6])]
    lags = [int(value) for value in supervised_config.get("lags", [1, 2, 3, 6, 12])]
    rolling_windows = [int(value) for value in supervised_config.get("rolling_windows", [3, 6, 12])]

    base = load_supervised_base(config)
    if supervised_config.get("include_umap_context", True):
        base = add_umap_context(
            base,
            date_column=date_column,
            umap_context_path=supervised_config.get(
                "umap_context_path", "data/processed/policy_umap_embedding.csv"
            ),
        )

    frame = base.copy()
    numeric_sources = [target_column, *feature_columns]
    engineered: dict[str, pd.Series | np.ndarray] = {}
    for column in numeric_sources:
        diff_column = frame[column].diff()
        engineered[f"{column}_diff_1"] = diff_column
        engineered[f"{column}_pct_change_1"] = frame[column].pct_change()
        for lag in lags:
            engineered[f"{column}_lag_{lag}"] = frame[column].shift(lag)
            engineered[f"{column}_diff_lag_{lag}"] = diff_column.shift(lag)

    return_col = f"{target_column}_log_return_1"
    for window in rolling_windows:
        engineered[f"{return_col}_rolling_mean_{window}"] = frame[return_col].rolling(window).mean()
        engineered[f"{return_col}_rolling_std_{window}"] = frame[return_col].rolling(window).std()

    dates = pd.to_datetime(frame[date_column])
    engineered["year"] = dates.dt.year
    engineered["month"] = dates.dt.month
    engineered["time_index"] = np.arange(len(frame))

    horizon_columns = {}
    for horizon in horizons:
        column = f"y_{target_column}_log_return_h{horizon}"
        engineered[column] = np.log(frame[target_column].shift(-horizon) / frame[target_column])
        horizon_columns[horizon] = column

    frame = pd.concat([frame, pd.DataFrame(engineered, index=frame.index)], axis=1)

    categorical_columns = ["executive_period", "policy_stance"]
    if "umap_cluster" in frame.columns:
        categorical_columns.append("umap_cluster")
    dummies = pd.get_dummies(frame[categorical_columns], prefix=categorical_columns, dummy_na=True, dtype=int)
    frame = pd.concat([frame, dummies], axis=1)

    excluded = {
        date_column,
        *horizon_columns.values(),
        "executive_period",
        "policy_stance",
        "umap_cluster",
    }
    feature_set = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]

    theory_base = supervised_config.get("grey_box", {}).get(
        "theory_features",
        ["trm", "tipm", "dtf", "precio_brent", "inflacion_sin_alimentos"],
    )
    stationary_tokens = ("_diff", "_pct_change", "policy_rate_change")
    theory_feature_columns = [
        column
        for column in feature_set
        if (
            any(column.startswith(f"{base}_") for base in theory_base)
            and any(token in column for token in stationary_tokens)
        )
        or column == "policy_rate_change"
    ]

    return SupervisedDataset(
        frame=frame.replace([np.inf, -np.inf], np.nan),
        feature_columns=feature_set,
        theory_feature_columns=theory_feature_columns,
        horizon_columns=horizon_columns,
    )


def build_supervised_dataset(config_path: str | Path = "configs/project.yaml") -> SupervisedDataset:
    """Build the time-series supervised learning matrix."""
    return build_supervised_dataset_from_config(load_config(config_path))


def split_train_test(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    date_column: str,
    train_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    complete = frame[[date_column, target_column, *feature_columns]].dropna(subset=[target_column])
    complete = complete.dropna(axis=0, how="any").sort_values(date_column)
    if len(complete) < 24:
        raise ValueError("Not enough complete observations for time-series train/test split.")

    split_at = int(len(complete) * train_size)
    split_at = min(max(split_at, 12), len(complete) - 6)

    train = complete.iloc[:split_at]
    test = complete.iloc[split_at:]
    return (
        train[feature_columns],
        test[feature_columns],
        train[target_column],
        test[target_column],
        train[date_column],
        test[date_column],
    )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true_array = y_true.to_numpy()
    return {
        "mae": float(mean_absolute_error(y_true_array, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred))),
        "r2": float(r2_score(y_true_array, y_pred)),
        "directional_accuracy": float((np.sign(y_true_array) == np.sign(y_pred)).mean()),
    }


def _rf_importance(model: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    forest = model.named_steps["model"]
    return (
        pd.DataFrame({"feature": feature_columns, "importance": forest.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _plot_predictions(predictions: pd.DataFrame, output_path: Path, horizon: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(predictions["fecha"], predictions["actual"], label="Actual", color="black", linewidth=2)
    for column in ["decision_tree", "random_forest", "grey_box_random_forest"]:
        ax.plot(predictions["fecha"], predictions[column], label=column, alpha=0.82)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_title(f"Gold log-return forecast, horizon {horizon}")
    ax.set_ylabel("Forward log return")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_importance(importance: pd.DataFrame, output_path: Path, title: str, top_n: int = 15) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"], color="#2f6f9f")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_random_forest_grey_box(config_path: str | Path = "configs/project.yaml") -> dict[str, Path]:
    """Train baselines, Random Forest and grey-box Random Forest for all horizons."""
    config = load_config(config_path)
    data_config = config["data"]
    supervised_config = config.get("supervised", {})
    rf_config = supervised_config.get("random_forest", {})
    tree_config = supervised_config.get("decision_tree", {})
    grey_config = supervised_config.get("grey_box", {})

    date_column = data_config["date_column"]
    train_size = float(supervised_config.get("train_size", 0.8))
    random_state = int(rf_config.get("random_state", 42))
    dataset = build_supervised_dataset(config_path)

    tables_dir = Path("reports/tables")
    figures_dir = Path("reports/figures")
    processed_dir = Path("data/processed")
    for directory in (tables_dir, figures_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_path = processed_dir / "random_forest_grey_box_dataset.csv"
    dataset.frame.to_csv(dataset_path, index=False)

    metrics_rows = []
    prediction_paths: dict[str, Path] = {"dataset": dataset_path}
    for horizon, target_column in dataset.horizon_columns.items():
        X_train, X_test, y_train, y_test, _, test_dates = split_train_test(
            dataset.frame,
            dataset.feature_columns,
            target_column,
            date_column,
            train_size,
        )

        models: dict[str, Any] = {
            "zero_return_baseline": ZeroReturnRegressor(),
            "decision_tree": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        DecisionTreeRegressor(
                            max_depth=tree_config.get("max_depth", 4),
                            min_samples_leaf=tree_config.get("min_samples_leaf", 5),
                            random_state=tree_config.get("random_state", random_state),
                        ),
                    ),
                ]
            ),
            "random_forest": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=rf_config.get("n_estimators", 500),
                            max_depth=rf_config.get("max_depth"),
                            min_samples_leaf=rf_config.get("min_samples_leaf", 3),
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "grey_box_random_forest": GreyBoxRandomForest(
                theory_features=dataset.theory_feature_columns,
                alpha_grid=grey_config.get("alpha_grid", [0.01, 0.1, 1.0, 10.0, 100.0]),
                random_state=random_state,
                n_estimators=rf_config.get("n_estimators", 500),
                max_depth=rf_config.get("max_depth"),
                min_samples_leaf=rf_config.get("min_samples_leaf", 3),
            ),
        }

        predictions = pd.DataFrame({date_column: test_dates.reset_index(drop=True), "actual": y_test.to_numpy()})
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions[model_name] = y_pred
            metrics = regression_metrics(y_test, y_pred)
            metrics_rows.append(
                {
                    "horizon": horizon,
                    "model": model_name,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    **metrics,
                }
            )

            if model_name == "random_forest":
                importance = _rf_importance(model, dataset.feature_columns)
                importance_path = tables_dir / f"random_forest_feature_importance_h{horizon}.csv"
                importance.to_csv(importance_path, index=False)
                _plot_importance(
                    importance,
                    figures_dir / f"random_forest_feature_importance_h{horizon}.png",
                    f"Random Forest feature importance, horizon {horizon}",
                )
                prediction_paths[f"rf_importance_h{horizon}"] = importance_path

                permutation = permutation_importance(
                    model,
                    X_test,
                    y_test,
                    scoring="neg_mean_absolute_error",
                    n_repeats=10,
                    random_state=random_state,
                    n_jobs=-1,
                )
                permutation_df = (
                    pd.DataFrame(
                        {
                            "feature": dataset.feature_columns,
                            "importance": permutation.importances_mean,
                            "importance_std": permutation.importances_std,
                        }
                    )
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )
                permutation_path = tables_dir / f"random_forest_permutation_importance_h{horizon}.csv"
                permutation_df.to_csv(permutation_path, index=False)
                prediction_paths[f"rf_permutation_h{horizon}"] = permutation_path

            if model_name == "grey_box_random_forest":
                residual_forest = model.residual_forest_
                residual_importance = _rf_importance(residual_forest, dataset.feature_columns)
                residual_path = tables_dir / f"grey_box_residual_importance_h{horizon}.csv"
                residual_importance.to_csv(residual_path, index=False)
                _plot_importance(
                    residual_importance,
                    figures_dir / f"grey_box_residual_importance_h{horizon}.png",
                    f"Grey-box residual RF importance, horizon {horizon}",
                )
                prediction_paths[f"grey_residual_importance_h{horizon}"] = residual_path

        prediction_path = tables_dir / f"random_forest_grey_box_predictions_h{horizon}.csv"
        predictions.to_csv(prediction_path, index=False)
        figure_path = figures_dir / f"random_forest_grey_box_predictions_h{horizon}.png"
        _plot_predictions(predictions, figure_path, horizon)
        prediction_paths[f"predictions_h{horizon}"] = prediction_path
        prediction_paths[f"predictions_figure_h{horizon}"] = figure_path

    metrics = pd.DataFrame(metrics_rows)
    metrics_path = tables_dir / "random_forest_grey_box_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    prediction_paths["metrics"] = metrics_path
    return prediction_paths
