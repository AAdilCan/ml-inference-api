"""Train XGBoost and LightGBM on UCI Adult income dataset, save artifacts."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
NUMERIC_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    log.info("fetching UCI Adult dataset from OpenML …")
    dataset = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df: pd.DataFrame = dataset.frame.copy()

    # target: <=50K → 0, >50K → 1
    target_col = dataset.target_names[0]
    y = (df[target_col].str.strip() == ">50K").astype(int)
    X = df.drop(columns=[target_col])
    return X, y


def preprocess(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    X = X.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        if col not in X.columns:
            continue
        le = LabelEncoder()
        X[col] = X[col].astype(str).str.strip()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    for col in NUMERIC_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X, encoders


def save_feature_schema(X: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> None:
    schema = {
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "feature_order": list(X.columns),
        "label_classes": {col: list(enc.classes_) for col, enc in encoders.items()},
    }
    path = MODELS_DIR / "feature_schema.json"
    path.write_text(json.dumps(schema, indent=2))
    log.info("feature schema saved → %s", path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
    }
    log.info(
        "%s — AUC: %.4f  F1: %.4f  Prec: %.4f  Rec: %.4f",
        name, metrics["roc_auc"], metrics["f1_macro"],
        metrics["precision"], metrics["recall"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    X, y = load_data()
    X_proc, encoders = preprocess(X)
    save_feature_schema(X_proc, encoders)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc.values, y.values, test_size=0.2, random_state=42, stratify=y
    )

    log.info("training XGBoost …")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_model.pkl")

    log.info("training LightGBM …")
    lgb_model = train_lgbm(X_train, y_train)
    lgb_metrics = evaluate(lgb_model, X_test, y_test, "LightGBM")
    joblib.dump(lgb_model, MODELS_DIR / "lightgbm_model.pkl")

    # save also the preprocessor encoders
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")

    results = {"xgboost": xgb_metrics, "lightgbm": lgb_metrics}
    results_path = MODELS_DIR / "eval_metrics.json"
    results_path.write_text(json.dumps(results, indent=2))
    log.info("metrics saved → %s", results_path)

    # pick best model (by AUC)
    best = max(results, key=lambda k: results[k]["roc_auc"])
    log.info("best model: %s (AUC=%.4f)", best, results[best]["roc_auc"])


if __name__ == "__main__":
    main()
