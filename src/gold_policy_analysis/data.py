from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_raw_path(path: str | Path, search_root: str | Path = ".") -> Path:
    """Resolve a raw Excel path locally or from Kaggle input folders."""
    candidate = Path(path)
    if candidate.exists():
        return candidate

    roots = [Path(search_root), Path("/kaggle/input"), Path("/kaggle/working")]
    for root in roots:
        if root.exists():
            matches = sorted(root.rglob("*.xlsx"))
            if matches:
                return matches[0]

    raise FileNotFoundError(f"Could not find raw Excel file from {path!s}.")


def slugify(value: object) -> str:
    """Convert spreadsheet headers into stable ASCII identifiers."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def load_raw_excel(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Read the raw Excel file and normalize column names."""
    df = pd.read_excel(resolve_raw_path(path), sheet_name=sheet_name)
    df = df.rename(columns={column: slugify(column) for column in df.columns})
    df = df.loc[:, [column for column in df.columns if column]]
    return df


def prepare_time_series(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Clean dates, coerce numeric columns and add minimal gold-price features."""
    required = [date_column, target_column, *feature_columns]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    out = df[required].copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce")
    out = out.dropna(subset=[date_column]).sort_values(date_column)

    numeric_columns = [target_column, *feature_columns]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=[target_column])
    out[f"{target_column}_return"] = out[target_column].pct_change()
    out[f"{target_column}_diff"] = out[target_column].diff()

    for column in feature_columns:
        out[f"{column}_diff"] = out[column].diff()

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.reset_index(drop=True)


def feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return a complete-case feature matrix for embedding and clustering."""
    matrix = df[feature_columns].copy()
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return matrix.dropna(axis=0, how="any")
