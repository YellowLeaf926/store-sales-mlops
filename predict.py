"""
predict.py — Inference Script
Loads a trained model and generates predictions for new input features.
Designed to be called by a Lambda function or ECS task for cloud deployment.

Usage:
    python predict.py --input data/features_test.parquet --output data/predictions.csv
    python predict.py --config config.yaml --input data/features_test.parquet
"""

import argparse
import logging
import tempfile

import boto3
import joblib
import numpy as np
import pandas as pd
import yaml

from monitor import publish_cloudwatch, setup_logging, write_metrics

log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fix_test_nans(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Fill NaN lag/rolling features with nearest available proxy."""
    X = df[features].copy()
    X["lag_7"] = X["lag_7"].fillna(X["lag_14"]).fillna(X["lag_28"])
    X["lag_14"] = X["lag_14"].fillna(X["lag_28"])
    X["rolling_mean_7"] = (
        X["rolling_mean_7"].fillna(X["rolling_mean_14"]).fillna(X["rolling_mean_28"])
    )
    X["rolling_mean_14"] = X["rolling_mean_14"].fillna(X["rolling_mean_28"])
    X["rolling_std_7"] = X["rolling_std_7"].fillna(0)
    return X


def run(
    model_path: str, input_path: str, output_path: str, monitoring_cfg: dict | None = None
) -> pd.DataFrame:
    log.info("Loading model from %s", model_path)
    if model_path.startswith("s3://"):
        bucket, key = model_path[5:].split("/", 1)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            boto3.client("s3").download_file(bucket, key, tmp.name)
            model = joblib.load(tmp.name)
    else:
        model = joblib.load(model_path)

    log.info("Loading input features from %s", input_path)
    feat = pd.read_parquet(input_path)

    NON_FEATURES = {"id", "date", "sales"}
    FEATURES = [c for c in feat.columns if c not in NON_FEATURES]
    log.info("Feature count: %d", len(FEATURES))

    X = fix_test_nans(feat, FEATURES)

    log.info("Running inference")
    log_preds = model.predict(X)
    preds = np.expm1(log_preds).clip(0)

    result = (
        pd.DataFrame({"id": feat["id"].values, "sales": preds})
        .sort_values("id")
        .reset_index(drop=True)
    )

    result.to_csv(output_path, index=False)
    log.info("Predictions saved to %s  (%d rows)", output_path, len(result))
    log.info(
        "Prediction stats — mean: %.2f  median: %.2f  max: %.2f",
        preds.mean(),
        np.median(preds),
        preds.max(),
    )

    # --- Monitoring ---
    mon = monitoring_cfg or {}
    metrics = {
        "run": "inference",
        "n_predictions": len(result),
        "mean_pred": round(float(preds.mean()), 4),
        "median_pred": round(float(np.median(preds)), 4),
        "max_pred": round(float(preds.max()), 4),
    }
    write_metrics(metrics, mon.get("metrics_file", "logs/metrics.json"))
    cw = mon.get("cloudwatch", {})
    if cw.get("enabled"):
        publish_cloudwatch(metrics, cw["namespace"], cw["region"])

    return result


def main():
    parser = argparse.ArgumentParser(description="Inference pipeline")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--model", default=None, help="Model .pkl path (overrides config)")
    parser.add_argument("--input", default=None, help="Input feature Parquet path")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_dir"], cfg["logging"]["level"])
    model_path = args.model or (cfg["paths"]["model_dir"] + "lgb_model.pkl")
    input_path = args.input or (cfg["paths"]["output_dir"] + "features_test.parquet")
    output_path = args.output or (cfg["paths"]["output_dir"] + "predictions.csv")

    run(model_path, input_path, output_path, cfg["monitoring"])


if __name__ == "__main__":
    main()