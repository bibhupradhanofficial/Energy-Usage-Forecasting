from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from metrics import evaluate_forecast
from prophet_forecasting import ensure_processed_splits
from model_artifacts import slugify, to_relpath, write_json


@dataclass(frozen=True)
class ArimaSpec:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]


def _coerce_hourly(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("ARIMA input series must have a DatetimeIndex.")
    s = s.sort_index()
    if s.index.freq is None:
        inferred = pd.infer_freq(s.index)
        if inferred is not None:
            s = s.asfreq(inferred)
    if s.index.freq is None:
        s = s.asfreq("h")
    s = s.ffill()
    return s


def select_sarimax_spec(y_train: pd.Series, seasonal_period: int = 24) -> ArimaSpec:
    candidates: list[ArimaSpec] = []
    for order in [(1, 1, 1), (2, 1, 2), (1, 0, 1), (2, 0, 2)]:
        for seasonal_order in [
            (1, 1, 1, seasonal_period),
            (0, 1, 1, seasonal_period),
            (1, 0, 1, seasonal_period),
            (0, 0, 0, 0),
        ]:
            candidates.append(ArimaSpec(order=order, seasonal_order=seasonal_order))

    best_spec = candidates[0]
    best_aic = np.inf

    for spec in candidates:
        try:
            model = SARIMAX(
                y_train,
                order=spec.order,
                seasonal_order=spec.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
        except Exception:
            continue

        if np.isfinite(res.aic) and res.aic < best_aic:
            best_aic = float(res.aic)
            best_spec = spec

    return best_spec


def plot_forecast_vs_actual(
    forecast: pd.DataFrame,
    actual: pd.Series,
    out_path: Path,
    title: str,
) -> None:
    df = forecast.copy().set_index("ds").sort_index()
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


def run_arima_forecast(
    target_col: str = "Global_active_power",
    seasonal_period: int = 24,
    max_train_points: int = 1500,
    output_dir: Path = Path("outputs"),
) -> tuple[dict[str, float], ArimaSpec]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    splits = ensure_processed_splits(target_col=target_col)
    y_train = _coerce_hourly(splits.train[target_col])
    y_test = _coerce_hourly(splits.test[target_col])

    if max_train_points > 0 and len(y_train) > max_train_points:
        y_train = y_train.iloc[-int(max_train_points) :]

    spec = select_sarimax_spec(y_train, seasonal_period=seasonal_period)
    model = SARIMAX(
        y_train,
        order=spec.order,
        seasonal_order=spec.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit_start = time.perf_counter()
    res = model.fit(disp=False)
    fit_seconds = float(time.perf_counter() - fit_start)

    steps = len(y_test)
    infer_start = time.perf_counter()
    fc = res.get_forecast(steps=steps)
    infer_seconds = float(time.perf_counter() - infer_start)
    mean = fc.predicted_mean
    conf = fc.conf_int()

    ds = pd.DatetimeIndex(y_test.index)
    forecast = pd.DataFrame(
        {
            "ds": ds,
            "yhat": mean.values,
            "yhat_lower": conf.iloc[:, 0].values,
            "yhat_upper": conf.iloc[:, 1].values,
        }
    )

    metrics = evaluate_forecast(y_test.reindex(ds).values, forecast["yhat"].values)

    merged = forecast.copy()
    merged["y_actual"] = y_test.reindex(ds).values
    merged.to_csv(output_dir / "arima_forecast.csv", index=False)

    pred_path = output_dir / "arima_h1_test_predictions.csv"
    pd.DataFrame({"datetime": ds, "y_true": y_test.reindex(ds).values, "y_pred": forecast["yhat"].values}).to_csv(
        pred_path, index=False
    )

    model_path = None
    try:
        model_path = output_dir / "arima_model.pkl"
        res.save(model_path)
    except Exception:
        model_path = None

    manifest = {
        "model_name": "SARIMAX",
        "model_family": "ARIMA",
        "horizon_hours": 1,
        "split": "test",
        "metrics": {"MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "MAPE": metrics["MAPE_%"]},
        "train_seconds": fit_seconds,
        "infer_seconds": infer_seconds,
        "predictions_csv": to_relpath(pred_path),
        "model_path": to_relpath(model_path) if model_path is not None else None,
        "extra": {"order": list(spec.order), "seasonal_order": list(spec.seasonal_order), "max_train_points": int(max_train_points)},
    }
    write_json(manifests_dir / f"arima_{slugify('sarimax')}_h1.json", manifest)

    plot_forecast_vs_actual(
        forecast=forecast,
        actual=y_test,
        out_path=output_dir / "arima_forecast_vs_actual.png",
        title=f"SARIMAX forecast vs actual (order={spec.order}, seasonal={spec.seasonal_order})",
    )

    return metrics, spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-col", default="Global_active_power")
    parser.add_argument("--seasonal-period", type=int, default=24)
    parser.add_argument("--max-train-points", type=int, default=1500)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    metrics, spec = run_arima_forecast(
        target_col=args.target_col,
        seasonal_period=args.seasonal_period,
        max_train_points=args.max_train_points,
        output_dir=Path(args.output_dir),
    )

    print(f"SARIMAX spec: order={spec.order}, seasonal_order={spec.seasonal_order}")
    print("ARIMA metrics on test:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
