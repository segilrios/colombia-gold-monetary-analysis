from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gold_policy_analysis.config import load_config
from gold_policy_analysis.supervised import (
    GreyBoxRandomForest,
    ZeroReturnRegressor,
    build_supervised_dataset_from_config,
    regression_metrics,
    split_train_test,
)


PRESETS: dict[str, dict[str, list[Any]]] = {
    "quick": {
        "lags": [[1, 2, 3, 6, 12]],
        "rolling_windows": [[3, 6, 12]],
        "n_estimators": [200],
        "max_depth": [None, 8],
        "min_samples_leaf": [1, 3],
    },
    "standard": {
        "lags": [[1, 2, 3, 6, 12], [1, 3, 6, 12, 18, 24]],
        "rolling_windows": [[3, 6, 12], [3, 6, 12, 24]],
        "n_estimators": [300, 600],
        "max_depth": [None, 6, 12],
        "min_samples_leaf": [1, 3, 5],
    },
    "exhaustive": {
        "lags": [[1, 2, 3, 6, 12], [1, 3, 6, 12, 18, 24], [1, 2, 3, 6, 9, 12, 18, 24]],
        "rolling_windows": [[3, 6, 12], [3, 6, 12, 24], [6, 12, 24]],
        "n_estimators": [300, 500, 1000],
        "max_depth": [None, 4, 8, 12],
        "min_samples_leaf": [1, 3, 5, 10],
    },
}


def _model_specs(
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    theory_features: list[str],
    alpha_grid: list[float],
) -> dict[str, Any]:
    return {
        "zero_return_baseline": ZeroReturnRegressor(),
        "decision_tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=max_depth if max_depth is not None else 6,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
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
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "grey_box_random_forest": GreyBoxRandomForest(
            theory_features=theory_features,
            alpha_grid=alpha_grid,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        ),
    }


def _write_rankings(results: pd.DataFrame, output_dir: Path) -> None:
    results_path = output_dir / "exhaustive_model_search_results.csv"
    best_path = output_dir / "exhaustive_model_search_best_by_horizon.csv"
    results.to_csv(results_path, index=False)
    best = (
        results.sort_values(["horizon", "rmse", "mae"])
        .groupby("horizon", as_index=False)
        .head(10)
        .reset_index(drop=True)
    )
    best.to_csv(best_path, index=False)


def run_search(config_path: str | Path, preset: str, max_combinations: int | None = None) -> Path:
    base_config = load_config(config_path)
    grid = PRESETS[preset]
    output_dir = Path("reports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    random_state = int(base_config.get("analysis", {}).get("random_state", 42))
    supervised = base_config.setdefault("supervised", {})
    grey_box = supervised.setdefault("grey_box", {})
    alpha_grid = grey_box.get("alpha_grid", [0.01, 0.1, 1.0, 10.0, 100.0])
    train_size = float(supervised.get("train_size", 0.8))
    date_column = base_config["data"]["date_column"]

    feature_grid = list(itertools.product(grid["lags"], grid["rolling_windows"]))
    model_grid = list(itertools.product(grid["n_estimators"], grid["max_depth"], grid["min_samples_leaf"]))
    total_combinations = len(feature_grid) * len(model_grid)
    if max_combinations is not None:
        total_combinations = min(total_combinations, max_combinations)

    rows: list[dict[str, Any]] = []
    completed = 0

    for lags, rolling_windows in feature_grid:
        config = copy.deepcopy(base_config)
        config["supervised"]["lags"] = list(lags)
        config["supervised"]["rolling_windows"] = list(rolling_windows)
        dataset = build_supervised_dataset_from_config(config)

        for n_estimators, max_depth, min_samples_leaf in model_grid:
            if max_combinations is not None and completed >= max_combinations:
                results = pd.DataFrame(rows)
                _write_rankings(results, output_dir)
                return output_dir / "exhaustive_model_search_results.csv"

            completed += 1
            print(
                f"[{completed}/{total_combinations}] "
                f"lags={lags} rolling={rolling_windows} "
                f"n_estimators={n_estimators} max_depth={max_depth} leaf={min_samples_leaf}",
                flush=True,
            )

            models = _model_specs(
                random_state=random_state,
                n_estimators=int(n_estimators),
                max_depth=max_depth,
                min_samples_leaf=int(min_samples_leaf),
                theory_features=dataset.theory_feature_columns,
                alpha_grid=alpha_grid,
            )

            for horizon, target_column in dataset.horizon_columns.items():
                X_train, X_test, y_train, y_test, _, _ = split_train_test(
                    dataset.frame,
                    dataset.feature_columns,
                    target_column,
                    date_column,
                    train_size,
                )

                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    prediction = model.predict(X_test)
                    metrics = regression_metrics(y_test, prediction)
                    rows.append(
                        {
                            "preset": preset,
                            "horizon": horizon,
                            "model": model_name,
                            "lags": " ".join(map(str, lags)),
                            "rolling_windows": " ".join(map(str, rolling_windows)),
                            "n_estimators": n_estimators,
                            "max_depth": "none" if max_depth is None else max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "n_features": len(dataset.feature_columns),
                            "n_theory_features": len(dataset.theory_feature_columns),
                            "n_train": len(X_train),
                            "n_test": len(X_test),
                            **metrics,
                        }
                    )

            results = pd.DataFrame(rows)
            _write_rankings(results, output_dir)

    return output_dir / "exhaustive_model_search_results.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exhaustive local model search.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="standard")
    parser.add_argument("--max-combinations", type=int, default=None)
    args = parser.parse_args()

    results_path = run_search(args.config, args.preset, args.max_combinations)
    print(f"results: {results_path}")
    print("best: reports/tables/exhaustive_model_search_best_by_horizon.csv")


if __name__ == "__main__":
    main()
