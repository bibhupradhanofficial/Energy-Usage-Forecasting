from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import io

import pandas as pd
import requests
import numpy as np


UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
    "household_power_consumption.zip"
)
DATA_FILENAME_IN_ZIP = "household_power_consumption.txt"


def download_and_extract_uci_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    txt_path = data_dir / DATA_FILENAME_IN_ZIP

    if txt_path.exists():
        return txt_path

    resp = requests.get(UCI_ZIP_URL, timeout=60)
    resp.raise_for_status()

    with ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extract(DATA_FILENAME_IN_ZIP, path=data_dir)

    return txt_path


def load_power_consumption(txt_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        txt_path,
        sep=";",
        na_values="?",
        low_memory=False,
    )


def add_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    out = df.copy()
    out["datetime"] = dt
    out = out.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
    out = out[out.index.notna()]
    return out


def build_hourly_target_dataset(
    df_raw: pd.DataFrame,
    target_col: str = "Global_active_power",
    min_obs_per_hour: int = 30,
    ffill_limit_hours: int = 24,
) -> pd.DataFrame:
    df = add_datetime_index(df_raw)
    s = pd.to_numeric(df[target_col], errors="coerce")
    s = s[~s.index.duplicated(keep="first")].sort_index()

    hourly_mean = s.resample("h").mean()
    hourly_count = s.resample("h").count()
    hourly = hourly_mean.where(hourly_count >= min_obs_per_hour)

    hourly = hourly.ffill(limit=ffill_limit_hours)
    hourly = hourly.dropna()

    out = hourly.to_frame(name=target_col)
    out.index.name = "datetime"
    return out


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or (train_frac + val_frac) >= 1:
        raise ValueError("train_frac and val_frac must be in (0, 1) and sum to < 1.")

    df = df.sort_index()
    n = len(df)
    if n < 10:
        raise ValueError(f"Not enough rows to split: {n}")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(f"Empty split produced: train={len(train)}, val={len(val)}, test={len(test)}")

    if not (train.index.max() < val.index.min() < test.index.min()):
        raise ValueError("Temporal split failed: index ordering is not strictly increasing across splits.")

    return train, val, test


def add_time_based_features(
    df_hourly: pd.DataFrame,
    target_col: str,
    dropna: bool = True,
) -> pd.DataFrame:
    if target_col not in df_hourly.columns:
        raise ValueError(f"target_col '{target_col}' not found in df_hourly columns.")

    df = df_hourly.sort_index().copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df_hourly must have a DatetimeIndex.")

    idx = df.index
    df["hour"] = idx.hour.astype("int16")
    df["day_of_week"] = idx.dayofweek.astype("int16")
    df["month"] = idx.month.astype("int16")
    df["quarter"] = idx.quarter.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_business_hour"] = (
        (df["day_of_week"] <= 4) & (df["hour"] >= 9) & (df["hour"] < 17)
    ).astype("int8")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    y = pd.to_numeric(df[target_col], errors="coerce")

    for h in (1, 2, 3, 6, 12, 24):
        df[f"lag_{h}h"] = y.shift(periods=h, freq="h")

    df["same_hour_yesterday"] = y.shift(periods=24, freq="h")
    df["same_hour_last_week"] = y.shift(periods=24 * 7, freq="h")

    y_hist = y.shift(periods=1, freq="h")
    for h in (6, 12, 24):
        df[f"roll_mean_{h}h"] = y_hist.rolling(f"{h}h", min_periods=h).mean()
        df[f"roll_std_{h}h"] = y_hist.rolling(f"{h}h", min_periods=h).std(ddof=0)

    feature_cols = [c for c in df.columns if c != target_col]
    if dropna:
        df = df.dropna(subset=feature_cols + [target_col])
    else:
        df[feature_cols] = df[feature_cols].ffill()
        df = df.dropna(subset=[target_col])

    return df


def feature_correlation_matrix(df_features: pd.DataFrame) -> pd.DataFrame:
    numeric = df_features.select_dtypes(include="number")
    corr = numeric.corr()
    return corr


def main() -> None:
    data_dir = Path("data")
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = download_and_extract_uci_dataset(data_dir)
    df_raw = load_power_consumption(txt_path)

    df_hourly = build_hourly_target_dataset(
        df_raw,
        target_col="Global_active_power",
        min_obs_per_hour=30,
        ffill_limit_hours=24,
    )

    target_col = "Global_active_power"
    df_features = add_time_based_features(df_hourly, target_col=target_col, dropna=True)

    train, val, test = temporal_train_val_test_split(df_features, train_frac=0.8, val_frac=0.1)

    df_hourly.to_csv(out_dir / "global_active_power_hourly.csv")
    df_features.to_csv(out_dir / "global_active_power_features.csv")
    train.to_csv(out_dir / "global_active_power_train.csv")
    val.to_csv(out_dir / "global_active_power_val.csv")
    test.to_csv(out_dir / "global_active_power_test.csv")

    corr = feature_correlation_matrix(df_features)
    corr.to_csv(out_dir / "feature_correlation_matrix.csv")

    target_corr = corr[target_col].drop(index=target_col).sort_values(ascending=False)

    print("Saved:")
    print(out_dir / "global_active_power_hourly.csv")
    print(out_dir / "global_active_power_features.csv")
    print(out_dir / "global_active_power_train.csv")
    print(out_dir / "global_active_power_val.csv")
    print(out_dir / "global_active_power_test.csv")
    print(out_dir / "feature_correlation_matrix.csv")
    print()
    print(f"Hourly rows: {len(df_hourly):,}")
    print(f"Feature rows (after NaN handling): {len(df_features):,}")
    print(f"Train/Val/Test: {len(train):,} / {len(val):,} / {len(test):,}")
    print("Date ranges:")
    print("train:", train.index.min(), "to", train.index.max())
    print("val:  ", val.index.min(), "to", val.index.max())
    print("test: ", test.index.min(), "to", test.index.max())
    print()
    print("Correlation matrix (head):")
    with pd.option_context("display.max_columns", 40, "display.width", 140):
        print(corr.round(3).iloc[:20, :20])
    print()
    print("Top correlations with target:")
    print(target_corr.head(15).round(3))


if __name__ == "__main__":
    main()
