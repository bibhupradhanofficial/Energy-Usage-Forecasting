from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import evaluate_forecast
from model_artifacts import slugify, to_relpath, write_json


def _import_prophet():
    try:
        from prophet import Prophet

        return Prophet
    except Exception as e:
        raise RuntimeError(
            "Prophet is not installed or failed to import. Install it with:\n"
            "  pip install prophet\n"
            "If you're on Windows, you may need to update pip first:\n"
            "  python -m pip install -U pip\n"
        ) from e


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def ensure_processed_splits(
    processed_dir: Path = Path("data") / "processed",
    target_col: str = "Global_active_power",
) -> SplitData:
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "global_active_power_train.csv"
    val_path = processed_dir / "global_active_power_val.csv"
    test_path = processed_dir / "global_active_power_test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        import prepare_time_series_data

        prepare_time_series_data.main()

    train = pd.read_csv(train_path, index_col=0, parse_dates=[0]).sort_index()
    val = pd.read_csv(val_path, index_col=0, parse_dates=[0]).sort_index()
    test = pd.read_csv(test_path, index_col=0, parse_dates=[0]).sort_index()

    for name, df in [("train", train), ("val", val), ("test", test)]:
        if target_col not in df.columns:
            raise ValueError(f"{name} split missing target column '{target_col}'.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{name} split index is not a DatetimeIndex.")

    return SplitData(train=train, val=val, test=test)


def prophet_regressors_from_features(df: pd.DataFrame) -> list[str]:
    candidates = [
        "hour",
        "day_of_week",
        "month",
        "quarter",
        "is_weekend",
        "is_business_hour",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]
    return [c for c in candidates if c in df.columns]


def to_prophet_frame(
    df: pd.DataFrame,
    target_col: str,
    regressors: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame({"ds": df.index, "y": pd.to_numeric(df[target_col], errors="coerce")})
    for r in regressors:
        out[r] = pd.to_numeric(df[r], errors="coerce")
    out = out.dropna().sort_values("ds")
    return out


def should_enable_yearly_seasonality(train_df: pd.DataFrame) -> bool:
    span_days = (train_df["ds"].max() - train_df["ds"].min()).days
    return span_days >= 365 * 2


def plot_forecast_vs_actual(
    forecast: pd.DataFrame,
    actual: pd.Series,
    out_path: Path,
    title: str,
) -> None:
    df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    df = df.set_index("ds").sort_index()
    aligned_actual = actual.reindex(df.index)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(aligned_actual.index, aligned_actual.values, label="Actual", linewidth=1)
    ax.plot(df.index, df["yhat"].values, label="Forecast (yhat)", linewidth=1)
    ax.fill_between(
        df.index,
        df["yhat_lower"].values,
        df["yhat_upper"].values,
        alpha=0.2,
        label="Uncertainty",
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Global_active_power")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_prophet_forecast(
    target_col: str = "Global_active_power",
    add_country_holidays: bool = True,
    country_name: str = "France",
    output_dir: Path = Path("outputs"),
) -> dict[str, float]:
    Prophet = _import_prophet()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    splits = ensure_processed_splits(target_col=target_col)
    regressors = prophet_regressors_from_features(splits.train)

    train_p = to_prophet_frame(splits.train, target_col=target_col, regressors=regressors)
    test_p = to_prophet_frame(splits.test, target_col=target_col, regressors=regressors)
    yearly = should_enable_yearly_seasonality(train_p)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=yearly,
    )
    if add_country_holidays:
        try:
            model.add_country_holidays(country_name=country_name)
        except Exception:
            pass

    for r in regressors:
        model.add_regressor(r)

    fit_start = time.perf_counter()
    model.fit(train_p)
    fit_seconds = float(time.perf_counter() - fit_start)

    future = test_p.drop(columns=["y"])
    infer_start = time.perf_counter()
    forecast = model.predict(future)
    infer_seconds = float(time.perf_counter() - infer_start)

    actual = test_p.set_index("ds")["y"].rename("y_actual")
    pred = forecast.set_index("ds")["yhat"]
    metrics = evaluate_forecast(actual.loc[pred.index].values, pred.values)

    merged = forecast.merge(actual.reset_index(), on="ds", how="left")
    merged.to_csv(output_dir / "prophet_forecast.csv", index=False)

    pred_path = output_dir / "prophet_h1_test_predictions.csv"
    pd.DataFrame(
        {"datetime": pred.index, "y_true": actual.loc[pred.index].values, "y_pred": pred.values}
    ).to_csv(pred_path, index=False)

    model_path = None
    try:
        import joblib

        model_path = output_dir / "prophet_model.joblib"
        joblib.dump(model, model_path)
    except Exception:
        model_path = None

    manifest = {
        "model_name": "Prophet",
        "model_family": "Prophet",
        "horizon_hours": 1,
        "split": "test",
        "metrics": {"MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "MAPE": metrics["MAPE_%"]},
        "train_seconds": fit_seconds,
        "infer_seconds": infer_seconds,
        "predictions_csv": to_relpath(pred_path),
        "model_path": to_relpath(model_path) if model_path is not None else None,
        "extra": {"regressors": regressors, "country_holidays": bool(add_country_holidays), "country_name": str(country_name)},
    }
    write_json(manifests_dir / f"prophet_{slugify('prophet')}_h1.json", manifest)

    plot_forecast_vs_actual(
        forecast=forecast,
        actual=actual,
        out_path=output_dir / "prophet_forecast_vs_actual.png",
        title="Prophet forecast vs actual",
    )

    fig = model.plot_components(forecast)
    fig.tight_layout()
    fig.savefig(output_dir / "prophet_components.png", dpi=150)
    plt.close(fig)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-col", default="Global_active_power")
    parser.add_argument("--no-holidays", action="store_true")
    parser.add_argument("--country", default="France")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    metrics = run_prophet_forecast(
        target_col=args.target_col,
        add_country_holidays=not args.no_holidays,
        country_name=args.country,
        output_dir=Path(args.output_dir),
    )

    with pd.option_context("display.width", 140):
        print("Prophet metrics on test:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
