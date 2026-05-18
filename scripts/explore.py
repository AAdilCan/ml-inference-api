"""Quick data exploration — class balance, feature distributions, missing values."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).parent.parent

CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
NUMERIC_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
]


def main() -> None:
    dataset = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df: pd.DataFrame = dataset.frame.copy()
    target_col = dataset.target_names[0]

    print("=== Dataset shape ===")
    print(df.shape)

    print("\n=== Target distribution ===")
    print(df[target_col].value_counts(normalize=True).round(3))

    print("\n=== Missing values ===")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "none")

    print("\n=== Numeric stats ===")
    print(df[NUMERIC_COLS].describe().round(2))

    print("\n=== Categorical cardinalities ===")
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            print(f"  {col}: {df[col].nunique()} unique")


if __name__ == "__main__":
    main()
