"""
preprocess.py — Step 2: Data Cleaning
Reads raw CSVs from data_dir, merges all tables, saves cleaned Parquet files.

Usage:
    python preprocess.py
    python preprocess.py --config config.yaml
"""

import argparse
import logging
import os

import pandas as pd
import yaml

from monitor import setup_logging

log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def clean_oil(oil: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill + backward-fill missing oil prices."""
    return (
        oil.set_index("date")
        .resample("D")
        .first()
        .ffill()
        .bfill()
        .reset_index()
        .rename(columns={"dcoilwtico": "oil_price"})
    )


def clean_holidays(holidays: pd.DataFrame):
    """Return geo-level holiday lookup tables (national / regional / local)."""
    hol = holidays[~holidays["transferred"]].copy()
    hol["type"] = hol["type"].replace("Transfer", "Holiday")

    nat = (
        hol[hol["locale"] == "National"][["date"]]
        .drop_duplicates()
        .assign(is_national_holiday=True)
    )
    reg = (
        hol[hol["locale"] == "Regional"][["date", "locale_name"]]
        .drop_duplicates()
        .rename(columns={"locale_name": "state"})
        .assign(is_regional_holiday=True)
    )
    loc = (
        hol[hol["locale"] == "Local"][["date", "locale_name"]]
        .drop_duplicates()
        .rename(columns={"locale_name": "city"})
        .assign(is_local_holiday=True)
    )
    return nat, reg, loc


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Fill missing (date, store_nbr) pairs with 0."""
    all_dates = pd.date_range(transactions["date"].min(), transactions["date"].max(), freq="D")
    all_stores = transactions["store_nbr"].unique()
    grid = pd.MultiIndex.from_product([all_dates, all_stores], names=["date", "store_nbr"])
    return transactions.set_index(["date", "store_nbr"]).reindex(grid, fill_value=0).reset_index()


def merge_all(df, stores, oil_clean, nat, reg, loc, tx_clean) -> pd.DataFrame:
    df = df.copy()
    df = df.merge(stores.rename(columns={"type": "store_type"}), on="store_nbr", how="left")
    df = df.merge(oil_clean[["date", "oil_price"]], on="date", how="left")

    df = df.merge(nat, on="date", how="left")
    df["is_national_holiday"] = df["is_national_holiday"].fillna(False)

    df = df.merge(reg, on=["date", "state"], how="left")
    df["is_regional_holiday"] = df["is_regional_holiday"].fillna(False)

    df = df.merge(loc, on=["date", "city"], how="left")
    df["is_local_holiday"] = df["is_local_holiday"].fillna(False)

    df["is_holiday"] = (
        df["is_national_holiday"] | df["is_regional_holiday"] | df["is_local_holiday"]
    )

    df = df.merge(tx_clean, on=["date", "store_nbr"], how="left")
    df["transactions"] = df["transactions"].fillna(0).astype(int)
    return df


def validate(train: pd.DataFrame, test: pd.DataFrame) -> None:
    assert (train["sales"] < 0).sum() == 0, "Negative sales found"
    assert train["oil_price"].isnull().sum() == 0, "Oil price has NaN"
    assert train["store_type"].isnull().sum() == 0, "store_type has NaN"
    assert train["is_national_holiday"].isnull().sum() == 0, "holiday flag has NaN"
    log.info("Validation passed")


def run(cfg: dict) -> None:
    data_dir = cfg["paths"]["data_dir"]
    output_dir = cfg["paths"]["output_dir"]
    if not output_dir.startswith("s3://"):
        os.makedirs(output_dir, exist_ok=True)

    log.info("Loading raw data from %s", data_dir)
    train = pd.read_csv(data_dir + "train.csv", parse_dates=["date"])
    test = pd.read_csv(data_dir + "test.csv", parse_dates=["date"])
    stores = pd.read_csv(data_dir + "stores.csv")
    oil = pd.read_csv(data_dir + "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(data_dir + "holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(data_dir + "transactions.csv", parse_dates=["date"])

    log.info("Cleaning oil prices (%d missing days)", oil["dcoilwtico"].isnull().sum())
    oil_clean = clean_oil(oil)

    log.info("Cleaning holidays")
    nat, reg, loc = clean_holidays(holidays)

    log.info("Cleaning transactions")
    tx_clean = clean_transactions(transactions)

    log.info("Merging tables")
    train_clean = merge_all(train, stores, oil_clean, nat, reg, loc, tx_clean)
    test_clean = merge_all(test, stores, oil_clean, nat, reg, loc, tx_clean)

    validate(train_clean, test_clean)

    train_out = output_dir + "train_cleaned.parquet"
    test_out = output_dir + "test_cleaned.parquet"
    train_clean.to_parquet(train_out, index=False)
    test_clean.to_parquet(test_out, index=False)

    log.info("Saved %s  (%d rows)", train_out, len(train_clean))
    log.info("Saved %s  (%d rows)", test_out, len(test_clean))


def main():
    parser = argparse.ArgumentParser(description="Data cleaning pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_dir"], cfg["logging"]["level"])
    run(cfg)


if __name__ == "__main__":
    main()