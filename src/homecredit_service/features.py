from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"
REQUIRED_DATA_FILES = (
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
)
TABLE_TEMPORAL_CUTOFFS: dict[str, tuple[str, ...]] = {
    "bureau": ("DAYS_CREDIT",),
    "bureau_balance": ("MONTHS_BALANCE",),
    "previous_application": ("DAYS_DECISION",),
    "POS_CASH_balance": ("MONTHS_BALANCE",),
    "installments_payments": ("DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"),
    "credit_card_balance": ("MONTHS_BALANCE",),
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace({0: np.nan})
    return result.replace([np.inf, -np.inf], np.nan)


def enrich_application_features(application_df: pd.DataFrame) -> pd.DataFrame:
    df = application_df.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["CREDIT_INCOME_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_ANNUITY_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    df["DAYS_EMPLOYED_PCT"] = safe_divide(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    return df


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    optimized = df.copy()
    for column in optimized.select_dtypes(include=["float64"]).columns:
        optimized[column] = optimized[column].astype("float32")
    for column in optimized.select_dtypes(include=["int64"]).columns:
        optimized[column] = pd.to_numeric(optimized[column], downcast="integer")
    return optimized


def read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    return downcast_numeric(df)


def filter_by_temporal_cutoff(df: pd.DataFrame, cutoff_columns: tuple[str, ...]) -> pd.DataFrame:
    existing_cutoffs = [column for column in cutoff_columns if column in df.columns]
    if not existing_cutoffs:
        return df

    keep_mask = pd.Series(True, index=df.index)
    for column in existing_cutoffs:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        keep_mask &= numeric_values.isna() | (numeric_values <= 0)
    return df.loc[keep_mask].copy()


def validate_data_dir(data_dir: Path) -> None:
    if not data_dir.exists():
        msg = f"Data directory does not exist: {data_dir}"
        raise FileNotFoundError(msg)

    missing_files = [name for name in REQUIRED_DATA_FILES if not (data_dir / name).exists()]
    if missing_files:
        missing = ", ".join(missing_files)
        msg = f"Missing required data files in {data_dir}: {missing}"
        raise FileNotFoundError(msg)


def aggregate_numeric(
    df: pd.DataFrame,
    group_key: str,
    prefix: str,
    drop_columns: Iterable[str] | None = None,
    stats: tuple[str, ...] = ("mean", "max", "min", "sum", "std"),
) -> pd.DataFrame:
    drops = set(drop_columns or [])
    drops.add(group_key)
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns if col not in drops
    ]

    grouped_size = df.groupby(group_key).size().to_frame(name=f"{prefix}_COUNT")
    if not numeric_cols:
        return grouped_size

    grouped = df.groupby(group_key)[numeric_cols].agg(stats)
    grouped.columns = [f"{prefix}_{name}_{agg}" for name, agg in grouped.columns]
    return grouped.join(grouped_size, how="left")


def build_auxiliary_aggregates(
    data_dir: Path, filter_keys: set[int] | None = None
) -> list[pd.DataFrame]:
    aggregates: list[pd.DataFrame] = []

    bureau = read_csv(data_dir / "bureau.csv")
    bureau = filter_by_temporal_cutoff(bureau, TABLE_TEMPORAL_CUTOFFS["bureau"])
    if filter_keys is not None:
        bureau = bureau[bureau[ID_COLUMN].isin(filter_keys)]
    bureau_agg = aggregate_numeric(bureau, ID_COLUMN, "BUREAU", drop_columns=["SK_ID_BUREAU"])
    aggregates.append(bureau_agg)

    bureau_map = bureau[["SK_ID_BUREAU", ID_COLUMN]].drop_duplicates()
    bureau_balance = read_csv(data_dir / "bureau_balance.csv")
    bureau_balance = filter_by_temporal_cutoff(
        bureau_balance, TABLE_TEMPORAL_CUTOFFS["bureau_balance"]
    )
    status_map = {"X": -1, "C": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    bureau_balance["STATUS_NUM"] = bureau_balance["STATUS"].map(status_map).astype("float32")
    bureau_balance = bureau_balance.merge(bureau_map, on="SK_ID_BUREAU", how="inner")
    bureau_balance = bureau_balance.drop(columns=["SK_ID_BUREAU", "STATUS"])
    bureau_balance_agg = aggregate_numeric(bureau_balance, ID_COLUMN, "BUREAU_BAL")
    aggregates.append(bureau_balance_agg)

    previous_app = read_csv(data_dir / "previous_application.csv")
    previous_app = filter_by_temporal_cutoff(
        previous_app, TABLE_TEMPORAL_CUTOFFS["previous_application"]
    )
    if filter_keys is not None:
        previous_app = previous_app[previous_app[ID_COLUMN].isin(filter_keys)]
    previous_agg = aggregate_numeric(previous_app, ID_COLUMN, "PREV", drop_columns=["SK_ID_PREV"])
    aggregates.append(previous_agg)

    pos_cash = read_csv(data_dir / "POS_CASH_balance.csv")
    pos_cash = filter_by_temporal_cutoff(pos_cash, TABLE_TEMPORAL_CUTOFFS["POS_CASH_balance"])
    if filter_keys is not None:
        pos_cash = pos_cash[pos_cash[ID_COLUMN].isin(filter_keys)]
    pos_agg = aggregate_numeric(pos_cash, ID_COLUMN, "POS", drop_columns=["SK_ID_PREV"])
    aggregates.append(pos_agg)

    installments = read_csv(data_dir / "installments_payments.csv")
    installments = filter_by_temporal_cutoff(
        installments, TABLE_TEMPORAL_CUTOFFS["installments_payments"]
    )
    if filter_keys is not None:
        installments = installments[installments[ID_COLUMN].isin(filter_keys)]
    installments_agg = aggregate_numeric(
        installments,
        ID_COLUMN,
        "INST",
        drop_columns=["SK_ID_PREV"],
    )
    aggregates.append(installments_agg)

    credit_card = read_csv(data_dir / "credit_card_balance.csv")
    credit_card = filter_by_temporal_cutoff(
        credit_card, TABLE_TEMPORAL_CUTOFFS["credit_card_balance"]
    )
    if filter_keys is not None:
        credit_card = credit_card[credit_card[ID_COLUMN].isin(filter_keys)]
    credit_card_agg = aggregate_numeric(credit_card, ID_COLUMN, "CC", drop_columns=["SK_ID_PREV"])
    aggregates.append(credit_card_agg)

    return aggregates


def merge_aggregates(base_df: pd.DataFrame, aggregates: list[pd.DataFrame]) -> pd.DataFrame:
    merged = base_df.copy()
    for aggregate_df in aggregates:
        merged = merged.merge(aggregate_df, how="left", left_on=ID_COLUMN, right_index=True)
    return merged


def build_training_frame(
    data_dir: Path,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    validate_data_dir(data_dir)
    application_train = enrich_application_features(read_csv(data_dir / "application_train.csv"))

    if sample_size is not None and sample_size < len(application_train):
        application_train = application_train.sample(n=sample_size, random_state=random_state)

    if application_train[ID_COLUMN].duplicated().any():
        msg = f"Duplicate {ID_COLUMN} values detected in application_train."
        raise ValueError(msg)

    key_filter = set(application_train[ID_COLUMN].astype(int).tolist())
    aggregates = build_auxiliary_aggregates(data_dir=data_dir, filter_keys=key_filter)
    training_frame = merge_aggregates(application_train, aggregates)

    return downcast_numeric(training_frame)
