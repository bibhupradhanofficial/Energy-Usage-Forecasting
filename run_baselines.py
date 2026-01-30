from __future__ import annotations

from pathlib import Path
import argparse
import time

import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forecasting_baselines import (
    arima_auto_forecast,
    evaluate_forecast,
    holt_winters_forecast,
    infer_step,
    moving_average_forecast,
    naive_forecast,
    seasonal_candidates_for_step,
    seasonal_naive_forecast,
    seasonal_strength,
)
from model_artifacts import slugify, to_relpath, write_json


def load_split(csv_path: Path, target_col: str) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in {csv_path.name}.")
    s = pd.to_numeric(df[target_col], errors="coerce").dropna()
    s.index = pd.DatetimeIndex(s.index)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s.rename(target_col)


def pick_seasonal_period(y_train: pd.Series) -> tuple[int | None, dict[str, float]]:
    step = infer_step(y_train.index)
    candidates = seasonal_candidates_for_step(step)
    strengths: dict[str, float] = {}
    for label, periods in candidates.items():
        strengths[label] = seasonal_strength(y_train, periods)
    best_label = None
    best_strength = -np.inf
    best_periods = None
    for label, strength in strengths.items():
        if np.isnan(strength):
            continue
        if strength > best_strength:
            best_strength = strength
            best_label = label
            best_periods = candidates[label]
    if best_label is None:
        return None, strengths
    return int(best_periods), strengths


def plot_best_model(
    y_test: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    out_path: Path,
) -> None:
    y_test, y_pred = y_test.align(y_pred, join="inner")

    plt.figure(figsize=(14, 6))
    plt.plot(y_test.index, y_test.to_numpy(), label="Actual", linewidth=1.2)
    plt.plot(y_pred.index, y_pred.to_numpy(), label=f"Predicted ({model_name})", linewidth=1.2)
    plt.title(f"Best baseline model: {model_name}")
    plt.xlabel("Time")
    plt.ylabel(y_test.name or "value")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_baselines(
    train_csv: Path,
    val_csv: Path | None,
    test_csv: Path,
    target_col: str,
    include_val_in_train: bool,
    out_dir: Path,
    outputs_dir: Path,
    max_arima_train_points: int = 5000,
    max_hw_train_points: int = 5000,
) -> tuple[pd.DataFrame, str, Path]:
    y_train = load_split(train_csv, target_col=target_col)
    y_test = load_split(test_csv, target_col=target_col)

    if include_val_in_train:
        if val_csv is None:
            raise ValueError("include_val_in_train=True but val_csv was not provided.")
        y_val = load_split(val_csv, target_col=target_col)
        y_fit = pd.concat([y_train, y_val]).sort_index()
    else:
        y_fit = y_train

    test_index = pd.DatetimeIndex(y_test.index)

    results = []
    timings: dict[str, dict[str, float]] = {}

    out_predictions_dir = Path(outputs_dir) / "baselines"
    out_predictions_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = Path(outputs_dir) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    y_pred = naive_forecast(y_fit, test_index)
    timings["Naive (last value)"] = {"train_seconds": 0.0, "infer_seconds": float(time.perf_counter() - t0)}
    results.append(evaluate_forecast("Naive (last value)", y_test, y_pred, metadata={}))

    seasonal_period, strengths = pick_seasonal_period(y_fit)
    strength_values = [v for v in strengths.values() if not np.isnan(v)]
    best_strength = max(strength_values) if strength_values else float("nan")
    if seasonal_period is not None:
        for label, periods in seasonal_candidates_for_step(infer_step(y_fit.index)).items():
            t0 = time.perf_counter()
            y_pred = seasonal_naive_forecast(y_fit, test_index, seasonal_periods=periods)
            timings[f"Seasonal naive ({label})"] = {"train_seconds": 0.0, "infer_seconds": float(time.perf_counter() - t0)}
            results.append(
                evaluate_forecast(
                    f"Seasonal naive ({label})",
                    y_test,
                    y_pred,
                    metadata={"seasonal_periods": periods, "seasonal_strength": strengths.get(label)},
                )
            )

    for window in (24, 48):
        t0 = time.perf_counter()
        y_pred = moving_average_forecast(y_fit, test_index, window=window)
        timings[f"Moving average ({window})"] = {"train_seconds": 0.0, "infer_seconds": float(time.perf_counter() - t0)}
        results.append(evaluate_forecast(f"Moving average ({window})", y_test, y_pred, metadata={"window": window}))

    y_fit_arima = y_fit.iloc[-int(max_arima_train_points) :] if max_arima_train_points > 0 else y_fit
    try:
        t0 = time.perf_counter()
        y_pred, meta = arima_auto_forecast(y_fit_arima, test_index, seasonal=False)
        timings["ARIMA (auto_arima)"] = {"train_seconds": float(time.perf_counter() - t0), "infer_seconds": 0.0}
        meta = dict(meta)
        meta["train_points_used"] = int(len(y_fit_arima))
        results.append(evaluate_forecast("ARIMA (auto_arima)", y_test, y_pred, metadata=meta))
    except Exception as e:
        y_pred = pd.Series(np.nan, index=test_index, name="ARIMA")
        timings["ARIMA (auto_arima) [failed]"] = {"train_seconds": float("nan"), "infer_seconds": float("nan")}
        results.append(
            evaluate_forecast(
                "ARIMA (auto_arima) [failed]",
                y_test,
                y_pred,
                metadata={"error": str(e), "train_points_used": int(len(y_fit_arima))},
            )
        )

    sarima_m = seasonal_period if seasonal_period is not None else None
    if sarima_m is not None and np.isfinite(best_strength) and best_strength >= 0.25:
        try:
            t0 = time.perf_counter()
            y_pred, meta = arima_auto_forecast(y_fit_arima, test_index, seasonal=True, m=sarima_m)
            timings["SARIMA (auto_arima)"] = {"train_seconds": float(time.perf_counter() - t0), "infer_seconds": 0.0}
            meta = dict(meta)
            meta["m"] = sarima_m
            meta["train_points_used"] = int(len(y_fit_arima))
            meta["seasonal_strength"] = float(best_strength)
            results.append(evaluate_forecast("SARIMA (auto_arima)", y_test, y_pred, metadata=meta))
        except Exception as e:
            y_pred = pd.Series(np.nan, index=test_index, name="SARIMA")
            timings["SARIMA (auto_arima) [failed]"] = {"train_seconds": float("nan"), "infer_seconds": float("nan")}
            results.append(
                evaluate_forecast(
                    "SARIMA (auto_arima) [failed]",
                    y_test,
                    y_pred,
                    metadata={
                        "error": str(e),
                        "m": sarima_m,
                        "train_points_used": int(len(y_fit_arima)),
                        "seasonal_strength": float(best_strength),
                    },
                )
            )
    else:
        y_pred = pd.Series(np.nan, index=test_index, name="SARIMA")
        timings["SARIMA (auto_arima) [skipped]"] = {"train_seconds": 0.0, "infer_seconds": 0.0}
        results.append(
            evaluate_forecast(
                "SARIMA (auto_arima) [skipped]",
                y_test,
                y_pred,
                metadata={"m": sarima_m, "seasonal_strength": float(best_strength), "threshold": 0.25},
            )
        )

    hw_seasonal = None
    hw_periods = None
    if seasonal_period is not None and np.isfinite(best_strength) and best_strength >= 0.25:
        hw_seasonal = "add"
        hw_periods = seasonal_period

    y_fit_hw = y_fit.iloc[-int(max_hw_train_points) :] if max_hw_train_points > 0 else y_fit
    t0 = time.perf_counter()
    y_pred, meta = holt_winters_forecast(
        y_fit_hw,
        test_index,
        seasonal_periods=hw_periods,
        seasonal=hw_seasonal,
        trend="add",
        damped_trend=True,
        use_brute=False,
    )
    timings["Exponential Smoothing (Holt-Winters)"] = {"train_seconds": float(time.perf_counter() - t0), "infer_seconds": 0.0}
    meta = dict(meta)
    meta["seasonal_strengths"] = strengths
    meta["train_points_used"] = int(len(y_fit_hw))
    results.append(evaluate_forecast("Exponential Smoothing (Holt-Winters)", y_test, y_pred, metadata=meta))

    for r in results:
        pred_path = out_predictions_dir / f"{slugify(r.model_name)}_h1_test_predictions.csv"
        y_true_aligned, y_pred_aligned = y_test.align(r.y_pred, join="inner")
        pd.DataFrame(
            {"datetime": y_true_aligned.index, "y_true": y_true_aligned.to_numpy(), "y_pred": y_pred_aligned.to_numpy()}
        ).to_csv(pred_path, index=False)
        t = timings.get(r.model_name, {"train_seconds": float("nan"), "infer_seconds": float("nan")})
        manifest = {
            "model_name": r.model_name,
            "model_family": "Baseline",
            "horizon_hours": 1,
            "split": "test",
            "metrics": dict(r.metrics),
            "train_seconds": float(t["train_seconds"]),
            "infer_seconds": float(t["infer_seconds"]),
            "predictions_csv": to_relpath(pred_path),
            "model_path": None,
            "extra": dict(r.metadata),
        }
        write_json(manifests_dir / f"baseline_{slugify(r.model_name)}_h1.json", manifest)

    comparison = pd.DataFrame(
        [
            {
                "Model": r.model_name,
                "MAE": r.metrics["MAE"],
                "RMSE": r.metrics["RMSE"],
                "MAPE": r.metrics["MAPE"],
                "Train_s": timings.get(r.model_name, {}).get("train_seconds", float("nan")),
                "Infer_s": timings.get(r.model_name, {}).get("infer_seconds", float("nan")),
            }
            for r in results
        ]
    ).sort_values(by="RMSE", ascending=True, kind="mergesort")

    best_model = str(comparison.iloc[0]["Model"])
    best_pred = None
    for r in results:
        if r.model_name == best_model:
            best_pred = r.y_pred
            break
    if best_pred is None:
        raise RuntimeError("Failed to identify best model prediction.")

    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "baseline_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    (out_predictions_dir / "baseline_comparison.csv").write_text(comparison.to_csv(index=False), encoding="utf-8")

    plot_path = out_dir / "best_baseline_forecast.png"
    plot_best_model(y_test, best_pred, best_model, plot_path)

    return comparison, best_model, plot_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--target-col", type=str, default="Global_active_power")
    parser.add_argument("--include-val-in-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-arima-train-points", type=int, default=1500)
    parser.add_argument("--max-hw-train-points", type=int, default=5000)
    args = parser.parse_args()

    processed_dir: Path = args.processed_dir
    train_csv = processed_dir / "global_active_power_train.csv"
    val_csv = processed_dir / "global_active_power_val.csv"
    test_csv = processed_dir / "global_active_power_test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            "Missing processed CSVs. Run prepare_time_series_data.py first to generate train/val/test files."
        )

    comparison, best_model, plot_path = run_baselines(
        train_csv=train_csv,
        val_csv=val_csv if val_csv.exists() else None,
        test_csv=test_csv,
        target_col=args.target_col,
        include_val_in_train=bool(args.include_val_in_train),
        out_dir=processed_dir,
        outputs_dir=Path("outputs"),
        max_arima_train_points=int(args.max_arima_train_points),
        max_hw_train_points=int(args.max_hw_train_points),
    )

    with pd.option_context("display.max_columns", 20, "display.width", 140):
        print(comparison.round(6).to_string(index=False))
    print()
    print("Best baseline model:", best_model)
    print("Saved comparison table:", processed_dir / "baseline_comparison.csv")
    print("Saved best-model plot:", plot_path)


if __name__ == "__main__":
    main()
