"""
train.py — Step 4 + 5: Baseline & LightGBM Training
Reads feature Parquet files, trains both models, saves model artifact and submissions.

Usage:
    python train.py
    python train.py --config config.yaml
"""

import argparse
import logging
import os
import tempfile

import boto3
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from monitor import publish_cloudwatch, setup_logging, write_metrics

log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def baseline_predict(tr: pd.DataFrame, target: pd.DataFrame, family_col: str) -> np.ndarray:
    """Predict mean sales per (store_nbr, family) from training split."""
    lookup = (
        tr.groupby(["store_nbr", family_col])["sales"]
        .mean()
        .reset_index()
        .rename(columns={"sales": "pred"})
    )
    return (
        target.merge(lookup, on=["store_nbr", family_col], how="left")["pred"]
        .fillna(0)
        .clip(lower=0)
        .values
    )


def train_lightgbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_cfg: dict,
) -> lgb.LGBMRegressor:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": model_cfg["n_estimators"],
        "learning_rate": model_cfg["learning_rate"],
        "num_leaves": model_cfg["num_leaves"],
        "min_child_samples": model_cfg["min_child_samples"],
        "feature_fraction": model_cfg["feature_fraction"],
        "bagging_fraction": model_cfg["bagging_fraction"],
        "bagging_freq": model_cfg["bagging_freq"],
        "lambda_l1": model_cfg["lambda_l1"],
        "lambda_l2": model_cfg["lambda_l2"],
        "random_state": model_cfg["random_state"],
        "n_jobs": model_cfg["n_jobs"],
        "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=model_cfg["early_stopping_rounds"], verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def fix_test_nans(test_feat: pd.DataFrame, features: list) -> pd.DataFrame:
    """Fill NaN lag/rolling features in test set with nearest available proxy."""
    df = test_feat[features].copy()
    df["lag_7"] = df["lag_7"].fillna(df["lag_14"]).fillna(df["lag_28"])
    df["lag_14"] = df["lag_14"].fillna(df["lag_28"])
    df["rolling_mean_7"] = (
        df["rolling_mean_7"].fillna(df["rolling_mean_14"]).fillna(df["rolling_mean_28"])
    )
    df["rolling_mean_14"] = df["rolling_mean_14"].fillna(df["rolling_mean_28"])
    df["rolling_std_7"] = df["rolling_std_7"].fillna(0)
    return df


def run(cfg: dict) -> None:
    data_dir = cfg["paths"]["data_dir"]
    model_dir = cfg["paths"]["model_dir"]
    output_dir = cfg["paths"]["output_dir"]
    tr_cfg = cfg["training"]
    model_cfg = cfg["model"]

    if not model_dir.startswith("s3://"):
        os.makedirs(model_dir, exist_ok=True)
    if not output_dir.startswith("s3://"):
        os.makedirs(output_dir, exist_ok=True)

    VAL_START = pd.Timestamp(tr_cfg["val_start"])
    VAL_END = pd.Timestamp(tr_cfg["val_end"])
    TR_END = pd.Timestamp(tr_cfg["tr_end"])

    log.info("Loading feature data")
    train_feat = pd.read_parquet(output_dir + "features_train.parquet")
    test_feat = pd.read_parquet(output_dir + "features_test.parquet")

    NON_FEATURES = {"id", "date", "sales"}
    FEATURES = [c for c in train_feat.columns if c not in NON_FEATURES]
    FAMILY_COL = "family_enc"

    log.info("Features: %d columns", len(FEATURES))

    # --- Train / val split ---
    tr = train_feat[train_feat["date"] <= TR_END]
    val = train_feat[(train_feat["date"] >= VAL_START) & (train_feat["date"] <= VAL_END)]
    log.info(
        "Train split: %s -> %s  (%d rows)",
        tr["date"].min().date(),
        tr["date"].max().date(),
        len(tr),
    )
    log.info(
        "Val split:   %s -> %s  (%d rows)",
        val["date"].min().date(),
        val["date"].max().date(),
        len(val),
    )

    # --- Baseline ---
    log.info("Running baseline model (mean per store+family)")
    baseline_pred = baseline_predict(tr, val, FAMILY_COL)
    baseline_score = rmsle(val["sales"].values, baseline_pred)
    log.info("Baseline RMSLE: %.4f", baseline_score)

    # --- LightGBM ---
    log.info("Training LightGBM")
    X_tr = tr[FEATURES]
    y_tr = np.log1p(tr["sales"])
    X_val = val[FEATURES]
    y_val = np.log1p(val["sales"])

    model = train_lightgbm(X_tr, y_tr, X_val, y_val, model_cfg)

    lgb_val_pred = np.expm1(model.predict(X_val, num_iteration=model.n_estimators)).clip(0)
    lgb_score = rmsle(val["sales"].values, lgb_val_pred)
    improvement = (baseline_score - lgb_score) / baseline_score * 100

    log.info("LightGBM RMSLE:  %.4f", lgb_score)
    log.info("Improvement over baseline: %.1f%%", improvement)

    # --- Retrain on full data ---
    log.info("Retraining on full data with n_estimators=%d", model.n_estimators)
    final_model = lgb.LGBMRegressor(
        n_estimators=model.n_estimators,
        **{k: v for k, v in model.get_params().items() if k != "n_estimators"},
    )
    final_model.fit(
        train_feat[FEATURES],
        np.log1p(train_feat["sales"]),
        callbacks=[lgb.log_evaluation(period=200)],
    )

    # --- Save model ---
    model_path = model_dir + "lgb_model.pkl"
    if model_dir.startswith("s3://"):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            joblib.dump(final_model, tmp.name)
            bucket, key = model_path[5:].split("/", 1)
            boto3.client("s3").upload_file(tmp.name, bucket, key)
    else:
        joblib.dump(final_model, model_path)
    log.info("Model saved: %s", model_path)

    # --- Test predictions ---
    log.info("Generating test predictions")
    X_test = fix_test_nans(test_feat, FEATURES)
    test_pred = np.expm1(final_model.predict(X_test)).clip(0)

    submission = (
        pd.DataFrame({"id": test_feat["id"].values, "sales": test_pred})
        .sort_values("id")
        .reset_index(drop=True)
    )
    sub_path = output_dir + "submission_lgb.csv"
    submission.to_csv(sub_path, index=False)
    log.info("Submission saved: %s  (%d rows)", sub_path, len(submission))

    # --- Summary ---
    log.info("=" * 45)
    log.info("Baseline RMSLE : %.4f", baseline_score)
    log.info("LightGBM RMSLE : %.4f  (%.1f%% improvement)", lgb_score, improvement)
    log.info("Model artifact : %s", model_path)
    log.info("=" * 45)

    # --- Monitoring ---
    metrics = {
        "run": "training",
        "baseline_rmsle": round(baseline_score, 4),
        "lgb_rmsle": round(lgb_score, 4),
        "improvement_pct": round(improvement, 2),
        "best_n_estimators": model.n_estimators,
    }
    write_metrics(metrics, cfg["monitoring"]["metrics_file"])
    cw = cfg["monitoring"]["cloudwatch"]
    if cw["enabled"]:
        publish_cloudwatch(metrics, cw["namespace"], cw["region"])


def main():
    parser = argparse.ArgumentParser(description="Model training pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_dir"], cfg["logging"]["level"])
    run(cfg)


if __name__ == "__main__":
    main()