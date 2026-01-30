from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from arima_forecasting import run_arima_forecast
from prophet_forecasting import run_prophet_forecast


def plot_overlay(
    df_prophet: pd.DataFrame | None,
    df_arima: pd.DataFrame | None,
    out_path: Path,
    title: str,
) -> None:
    actual_series = None
    if df_prophet is not None and "y_actual" in df_prophet.columns:
        actual_series = df_prophet.set_index("ds")["y_actual"]
    if actual_series is None and df_arima is not None and "y_actual" in df_arima.columns:
        actual_series = df_arima.set_index("ds")["y_actual"]

    fig, ax = plt.subplots(figsize=(14, 5))
    if actual_series is not None:
        ax.plot(actual_series.index, actual_series.values, label="Actual", linewidth=1)

    if df_prophet is not None:
        p = df_prophet.set_index("ds")["yhat"].rename("Prophet")
        ax.plot(p.index, p.values, label="Prophet", linewidth=1)

    if df_arima is not None:
        a = df_arima.set_index("ds")["yhat"].rename("ARIMA")
        ax.plot(a.index, a.values, label="ARIMA", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Global_active_power")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-col", default="Global_active_power")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seasonal-period", type=int, default=24)
    parser.add_argument("--max-arima-train-points", type=int, default=1500)
    parser.add_argument("--no-holidays", action="store_true")
    parser.add_argument("--country", default="France")
    args = parser.parse_args()

    root_out = Path(args.output_dir)
    prophet_out = root_out / "prophet"
    arima_out = root_out / "arima"
    compare_out = root_out / "compare"
    compare_out.mkdir(parents=True, exist_ok=True)

    prophet_metrics: dict[str, float] | None = None
    try:
        prophet_metrics = run_prophet_forecast(
            target_col=args.target_col,
            add_country_holidays=not args.no_holidays,
            country_name=args.country,
            output_dir=prophet_out,
        )
    except Exception as e:
        print(f"Prophet failed: {e}")

    arima_metrics, arima_spec = run_arima_forecast(
        target_col=args.target_col,
        seasonal_period=args.seasonal_period,
        max_train_points=args.max_arima_train_points,
        output_dir=arima_out,
    )

    rows: list[dict[str, object]] = []
    if prophet_metrics is not None:
        rows.append({"model": "Prophet", **prophet_metrics})
    rows.append({"model": f"ARIMA (SARIMAX {arima_spec.order} x {arima_spec.seasonal_order})", **arima_metrics})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(compare_out / "metrics_comparison.csv", index=False)

    df_prophet = None
    df_arima = None
    prophet_csv = prophet_out / "prophet_forecast.csv"
    arima_csv = arima_out / "arima_forecast.csv"
    if prophet_csv.exists():
        df_prophet = pd.read_csv(prophet_csv, parse_dates=["ds"])
    if arima_csv.exists():
        df_arima = pd.read_csv(arima_csv, parse_dates=["ds"])

    plot_overlay(
        df_prophet=df_prophet,
        df_arima=df_arima,
        out_path=compare_out / "forecast_overlay.png",
        title="Forecast overlay: Prophet vs ARIMA",
    )

    with pd.option_context("display.width", 140):
        print("Metrics comparison (test):")
        print(metrics_df)
        print()
        print("Saved:")
        print(compare_out / "metrics_comparison.csv")
        print(compare_out / "forecast_overlay.png")


if __name__ == "__main__":
    main()
