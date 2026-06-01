"""
features.py — Step 3: Feature Engineering
Reads cleaned Parquet files, builds all features, saves feature Parquet files.

Usage:
    python features.py
    python features.py --config config.yaml
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import yaml

from monitor import setup_logging

log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_on_promo"] = (df["onpromotion"] > 0).astype(int)
    df["log_onpromotion"] = np.log1p(df["onpromotion"])
    return df


def add_oil_rolling(df: pd.DataFrame) -> pd.DataFrame:
    oil_daily = df[["date", "oil_price"]].drop_duplicates("date").sort_values("date").copy()
    oil_daily["oil_ma7"] = oil_daily["oil_price"].rolling(7, min_periods=1).mean()
    oil_daily["oil_ma28"] = oil_daily["oil_price"].rolling(28, min_periods=1).mean()
    return df.merge(oil_daily[["date", "oil_ma7", "oil_ma28"]], on="date", how="left")


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["store_nbr", "family"])["sales"]
    for lag in [7, 14, 28, 35]:
        df[f"lag_{lag}"] = grp.shift(lag)
    return df


def _rolling_agg(series: pd.Series, window: int, func: str) -> pd.Series:
    shifted = series.shift(1)
    if func == "mean":
        return shifted.rolling(window, min_periods=1).mean()
    elif func == "std":
        return shifted.rolling(window, min_periods=1).std()
    elif func == "max":
        return shifted.rolling(window, min_periods=1).max()


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["store_nbr", "family"])["sales"]
    df["rolling_mean_7"] = grp.transform(lambda x: _rolling_agg(x, 7, "mean"))
    df["rolling_mean_14"] = grp.transform(lambda x: _rolling_agg(x, 14, "mean"))
    df["rolling_mean_28"] = grp.transform(lambda x: _rolling_agg(x, 28, "mean"))
    df["rolling_std_7"] = grp.transform(lambda x: _rolling_agg(x, 7, "std"))
    df["rolling_max_28"] = grp.transform(lambda x: _rolling_agg(x, 28, "max"))
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["family", "store_type", "city", "state"]:
        df[f"{col}_enc"], _ = pd.factorize(df[col])
    for col in ["is_national_holiday", "is_regional_holiday", "is_local_holiday", "is_holiday"]:
        df[col] = df[col].astype(int)
    return df


def run(cfg: dict) -> None:
    data_dir = cfg["paths"]["data_dir"]
    output_dir = cfg["paths"]["output_dir"]
    if not output_dir.startswith("s3://"):
        os.makedirs(output_dir, exist_ok=True)

    log.info("Loading cleaned data")
    train = pd.read_parquet(data_dir + "train_cleaned.parquet")
    test = pd.read_parquet(data_dir + "test_cleaned.parquet")

    log.info("Combining train + test for lag computation")
    train["is_train"] = True
    test["is_train"] = False
    test["sales"] = np.nan

    full = (
        pd.concat([train, test], sort=False)
        .sort_values(["store_nbr", "family", "date"])
        .reset_index(drop=True)
    )

    log.info("Adding date features")
    full = add_date_features(full)

    log.info("Adding promotion features")
    full = add_promo_features(full)

    log.info("Adding oil rolling features")
    full = add_oil_rolling(full)

    log.info("Adding lag features")
    full = add_lag_features(full)

    log.info("Adding rolling statistics")
    full = add_rolling_features(full)

    log.info("Encoding categorical features")
    full = encode_categoricals(full)

    drop_cols = ["city", "state", "family", "store_type", "is_train"]
    full = full.drop(columns=[c for c in drop_cols if c in full.columns])

    train_feat = full[full["sales"].notna()].copy()
    test_feat = full[full["sales"].isna()].drop(columns=["sales"]).copy()

    train_out = output_dir + "features_train.parquet"
    test_out = output_dir + "features_test.parquet"
    train_feat.to_parquet(train_out, index=False)
    test_feat.to_parquet(test_out, index=False)

    log.info("Saved %s  (%d rows x %d cols)", train_out, *train_feat.shape)
    log.info("Saved %s  (%d rows x %d cols)", test_out, *test_feat.shape)


def main():
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_dir"], cfg["logging"]["level"])
    run(cfg)


if __name__ == "__main__":
    main()