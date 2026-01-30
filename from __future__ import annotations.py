from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import io
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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

    print(f"Downloading from: {UCI_ZIP_URL}")
    resp = requests.get(UCI_ZIP_URL, timeout=60)
    resp.raise_for_status()

    with ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extract(DATA_FILENAME_IN_ZIP, path=data_dir)

    return txt_path


def load_power_consumption(txt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        txt_path,
        sep=";",
        na_values="?",                 # UCI uses '?' to denote missing
        low_memory=False,
    )
    return df


def inspect_dataframe(df: pd.DataFrame) -> None:
    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- dtypes ---")
    print(df.dtypes)

    print("\n--- Basic statistics (numeric) ---")
    print(df.describe())


def missing_values_report(df: pd.DataFrame) -> pd.Series:
    missing = df.isna().sum().sort_values(ascending=False)
    print("\n--- Missing values per column ---")
    print(missing[missing > 0] if (missing > 0).any() else "No missing values detected.")
    return missing


def explain_columns() -> None:
    print(
        """
--- Column meanings (UCI 'Individual household electric power consumption') ---
Date: Date in format dd/mm/yyyy
Time: Time in format hh:mm:ss

Global_active_power: Household global minute-averaged active power (kilowatt)
Global_reactive_power: Household global minute-averaged reactive power (kilowatt)
Voltage: Minute-averaged voltage (volt)
Global_intensity: Household global minute-averaged current intensity (ampere)

Sub_metering_1: Energy sub-metering No. 1 (watt-hour of active energy)
                (kitchen: dishwasher, oven, microwave)
Sub_metering_2: Energy sub-metering No. 2 (watt-hour of active energy)
                (laundry room: washing machine, dryer, refrigerator, light)
Sub_metering_3: Energy sub-metering No. 3 (watt-hour of active energy)
                (electric water-heater and air-conditioner)
"""
    )


def add_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    # Combine Date + Time into a single datetime, then set as index
    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.copy()
    df["datetime"] = dt
    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()

    # Optional: remove rows where datetime parsing failed
    df = df[df.index.notna()]

    return df


def get_power_series(df: pd.DataFrame, column: str = "Global_active_power") -> pd.Series:
    s = pd.to_numeric(df[column], errors="coerce")
    s = s.dropna()
    s = s[~s.index.duplicated(keep="first")]
    s = s.sort_index()
    s.name = column
    return s


def plot_overall_trend(s: pd.Series) -> None:
    daily = s.resample("D").mean()
    smooth = daily.rolling(window=7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.index, daily.values, linewidth=0.8, alpha=0.65, label="Daily mean")
    ax.plot(smooth.index, smooth.values, linewidth=2.0, label="7-day rolling mean")
    ax.set_title("Overall power consumption trend (daily mean)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{s.name} (kW)")
    ax.legend()
    fig.tight_layout()

    print("\nEDA: Overall trend")
    print("- Daily mean highlights long-term changes and shifts in baseline usage.")
    print("- 7-day rolling mean smooths noise and makes gradual trend changes clearer.")


def plot_daily_weekly_monthly_patterns(s: pd.Series) -> None:
    hourly_mean = s.groupby(s.index.hour).mean().rename("mean")

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = s.groupby(s.index.day_name()).mean().reindex(dow_order).rename("mean")

    month_order = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    month_seasonal = (
        s.groupby(s.index.month_name().str.slice(0, 3))
        .mean()
        .reindex(month_order)
        .rename("mean")
    )
    monthly = s.resample("MS").mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sns.lineplot(x=hourly_mean.index, y=hourly_mean.values, ax=axes[0, 0])
    axes[0, 0].set_title("Daily pattern: average consumption by hour of day")
    axes[0, 0].set_xlabel("Hour of day")
    axes[0, 0].set_ylabel(f"{s.name} (kW)")

    sns.barplot(x=dow.index, y=dow.values, ax=axes[0, 1])
    axes[0, 1].set_title("Weekly pattern: average consumption by day of week")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel(f"{s.name} (kW)")
    axes[0, 1].tick_params(axis="x", rotation=30)

    sns.barplot(x=month_seasonal.index, y=month_seasonal.values, ax=axes[1, 0])
    axes[1, 0].set_title("Seasonality: average consumption by month of year")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel(f"{s.name} (kW)")

    axes[1, 1].plot(monthly.index, monthly.values, linewidth=1.2)
    axes[1, 1].set_title("Monthly pattern over time: monthly mean consumption")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel(f"{s.name} (kW)")

    fig.tight_layout()

    print("\nEDA: Daily/weekly/monthly patterns")
    print("- Hour-of-day profile reveals typical peaks (e.g., mornings/evenings) and low-usage periods.")
    print("- Day-of-week differences suggest workday vs weekend behavior changes.")
    print("- Month-of-year averages indicate seasonality (heating/cooling effects).")
    print("- Monthly mean over time shows slower regime shifts not visible in hourly variation.")


def detect_anomalies_rolling_zscore(
    s: pd.Series,
    freq: str = "H",
    window: int = 24 * 14,
    z_thresh: float = 4.0,
) -> pd.DataFrame:
    x = s.resample(freq).mean().dropna()
    med = x.rolling(window=window, min_periods=max(10, window // 10)).median()
    mad = (x - med).abs().rolling(window=window, min_periods=max(10, window // 10)).median()
    robust_z = 0.6745 * (x - med) / mad.replace(0, pd.NA)
    anomalies = robust_z.abs() >= z_thresh
    return pd.DataFrame(
        {"value": x, "rolling_median": med, "robust_z": robust_z, "is_anomaly": anomalies}
    )


def plot_anomalies(s: pd.Series) -> None:
    df_anom = detect_anomalies_rolling_zscore(s, freq="H")
    anomalies = df_anom[df_anom["is_anomaly"]]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_anom.index, df_anom["value"], linewidth=0.8, alpha=0.7, label="Hourly mean")
    ax.plot(df_anom.index, df_anom["rolling_median"], linewidth=1.6, label="Rolling median")
    ax.scatter(
        anomalies.index,
        anomalies["value"],
        s=18,
        color="crimson",
        alpha=0.85,
        label=f"Anomalies (|z| ≥ 4.0): {len(anomalies)}",
    )
    ax.set_title("Anomaly detection on hourly mean (robust rolling z-score)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{s.name} (kW)")
    ax.legend()
    fig.tight_layout()

    fig2, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(df_anom["robust_z"].dropna(), bins=80, ax=axes[0])
    axes[0].set_title("Distribution of robust z-scores")
    axes[0].set_xlabel("Robust z-score")
    axes[0].set_ylabel("Count")

    sns.boxplot(y=df_anom["value"].dropna(), ax=axes[1])
    axes[1].set_title("Boxplot of hourly mean consumption")
    axes[1].set_ylabel(f"{s.name} (kW)")
    fig2.tight_layout()

    print("\nEDA: Anomalies/outliers")
    print("- Red points are unusually high/low hours relative to local behavior (rolling median).")
    print("- Spikes can indicate unusual events, sensor glitches, outages, or behavioral changes.")
    print("- Boxplot and z-score distribution show tail heaviness and potential extreme values.")


def adf_test_and_print(s: pd.Series) -> dict:
    x = s.resample("H").mean().dropna()
    result = adfuller(x.values, autolag="AIC")
    test_stat, p_value, used_lag, n_obs, crit_values, _ = result

    out = {
        "test_stat": float(test_stat),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": {k: float(v) for k, v in crit_values.items()},
    }

    print("\nEDA: Stationarity (Augmented Dickey-Fuller on hourly mean)")
    print(f"- ADF statistic: {out['test_stat']:.4f}")
    print(f"- p-value: {out['p_value']:.6f}")
    print(f"- used lag: {out['used_lag']}, observations: {out['n_obs']}")
    print(f"- critical values: {out['critical_values']}")
    if out["p_value"] < 0.05:
        print("- Interpretation: reject unit root (series is likely stationary).")
    else:
        print("- Interpretation: fail to reject unit root (series is likely non-stationary).")

    return out


def plot_acf_pacf_for_series(s: pd.Series, title_prefix: str, lags: int = 168) -> None:
    x = s.resample("H").mean().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(x, lags=lags, ax=axes[0])
    axes[0].set_title(f"{title_prefix} ACF (hourly mean)")

    plot_pacf(x, lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"{title_prefix} PACF (hourly mean)")
    fig.tight_layout()

    print("\nEDA: ACF/PACF")
    print("- ACF shows repeating autocorrelation peaks at seasonal lags (daily/weekly cycles).")
    print("- PACF highlights direct lag relationships useful for AR model order intuition.")


def run_eda(df: pd.DataFrame, target_col: str = "Global_active_power") -> None:
    sns.set_theme(style="whitegrid")
    s = get_power_series(df, target_col)
    plot_overall_trend(s)
    plot_daily_weekly_monthly_patterns(s)
    plot_anomalies(s)
    adf = adf_test_and_print(s)
    plot_acf_pacf_for_series(s, title_prefix="Raw")

    if adf["p_value"] >= 0.05:
        s_diff = s.resample("H").mean().dropna().diff().dropna()
        plot_acf_pacf_for_series(s_diff, title_prefix="1st-differenced")
        print("- Differenced ACF/PACF helps if the raw series is non-stationary.")

    plt.show()


def main() -> None:
    data_dir = Path("data")  # change if you want another folder
    txt_path = download_and_extract_uci_dataset(data_dir)

    df_raw = load_power_consumption(txt_path)

    inspect_dataframe(df_raw)
    missing_values_report(df_raw)
    explain_columns()

    df = add_datetime_index(df_raw)

    print("\n--- After datetime index conversion ---")
    print(df.head())
    print("\nIndex dtype:", df.index.dtype)
    print("Date range:", df.index.min(), "to", df.index.max())

    run_eda(df, target_col="Global_active_power")


if __name__ == "__main__":
    main()
