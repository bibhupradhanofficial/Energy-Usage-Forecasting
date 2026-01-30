from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from metrics import evaluate_forecast
from model_artifacts import slugify, to_relpath, write_json


@dataclass(frozen=True)
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray
    y_index: pd.DatetimeIndex


def _load_indexed_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def _load_split_bounds(processed_dir: Path) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    bounds = {}
    for name in ("train", "val", "test"):
        p = processed_dir / f"global_active_power_{name}.csv"
        if not p.exists():
            continue
        df = _load_indexed_csv(p)
        if len(df.index) == 0:
            continue
        bounds[name] = (pd.Timestamp(df.index.min()), pd.Timestamp(df.index.max()))
    return bounds


def _infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError("Need at least two timestamps to infer step.")
    diffs = pd.Series(index[1:] - index[:-1]).dropna()
    if diffs.empty:
        raise ValueError("Could not infer step from index.")
    return diffs.mode().iloc[0]


def _build_sequences(
    features: np.ndarray,
    target: np.ndarray,
    timestamps: pd.DatetimeIndex,
    lookback: int,
    step: pd.Timedelta,
    horizon_hours: int,
) -> SequenceDataset:
    if features.ndim != 2:
        raise ValueError("features must be 2D array of shape [time, features].")
    if target.ndim != 1:
        raise ValueError("target must be 1D array of shape [time].")
    if len(features) != len(target) or len(target) != len(timestamps):
        raise ValueError("features, target, and timestamps must have the same length.")
    if lookback < 1:
        raise ValueError("lookback must be >= 1.")
    horizon_hours = int(horizon_hours)
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be >= 1.")

    ts_ns = timestamps.view("i8")
    diffs = np.diff(ts_ns)
    step_ns = int(pd.Timedelta(step).value)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    y_ts: list[pd.Timestamp] = []

    for end in range(int(lookback), len(target) - int(horizon_hours) + 1):
        start = end - int(lookback)
        if start < 0:
            continue
        if not np.all(diffs[start:end] == step_ns):
            continue
        X_list.append(features[start:end])
        target_pos = end + int(horizon_hours) - 1
        y_list.append(float(target[target_pos]))
        y_ts.append(pd.Timestamp(timestamps[target_pos]))

    if not X_list:
        raise RuntimeError("No sequences were created. Check lookback and time step continuity.")

    X = np.stack(X_list, axis=0).astype("float32")
    y = np.asarray(y_list, dtype="float32")
    y_index = pd.DatetimeIndex(y_ts)
    return SequenceDataset(X=X, y=y, y_index=y_index)


def _subset_sequences(
    ds: SequenceDataset,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> SequenceDataset:
    mask = (ds.y_index >= start) & (ds.y_index <= end)
    X = ds.X[mask]
    y = ds.y[mask]
    y_index = ds.y_index[mask]
    if len(y_index) == 0:
        raise RuntimeError(f"No sequences fall in range [{start}, {end}].")
    return SequenceDataset(X=X, y=y, y_index=y_index)


def _plot_history(history: dict[str, list[float]], out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    if "loss" in history:
        plt.plot(history["loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    plt.title("Training history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: pd.DatetimeIndex,
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(index, y_true, label="Actual", linewidth=1)
    plt.plot(index, y_pred, label="Predicted", linewidth=1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Global_active_power")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _try_load_other_metrics(processed_dir: Path, outputs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    baseline_path = processed_dir / "baseline_comparison.csv"
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        if {"Model", "MAE", "RMSE", "MAPE"}.issubset(df.columns) and not df.empty:
            best = df.sort_values(by="RMSE", ascending=True, kind="mergesort").iloc[0]
            rows.append({"Model": f"Baseline: {best['Model']}", "MAE": best["MAE"], "RMSE": best["RMSE"], "MAPE": best["MAPE"]})

    ml_path = outputs_dir / "ml" / "ml_model_comparison.csv"
    if ml_path.exists():
        df = pd.read_csv(ml_path)
        if {"Model", "MAE", "RMSE", "MAPE"}.issubset(df.columns) and not df.empty:
            best = df.sort_values(by="RMSE", ascending=True, kind="mergesort").iloc[0]
            rows.append({"Model": f"ML: {best['Model']}", "MAE": best["MAE"], "RMSE": best["RMSE"], "MAPE": best["MAPE"]})

    compare_path = outputs_dir / "compare" / "metrics_comparison.csv"
    if compare_path.exists():
        df = pd.read_csv(compare_path)
        if {"model", "MAE", "RMSE", "MAPE"}.issubset(df.columns):
            for _, r in df.iterrows():
                rows.append({"Model": str(r["model"]), "MAE": float(r["MAE"]), "RMSE": float(r["RMSE"]), "MAPE": float(r["MAPE"])})

    if not rows:
        return pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE"])
    return pd.DataFrame(rows)


def run_lstm(
    processed_dir: Path,
    outputs_dir: Path,
    target_col: str,
    lookback_hours: int,
    horizon_hours: int,
    lstm_units: list[int],
    dropout: float,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    include_time_features: bool,
    random_state: int,
) -> tuple[dict[str, float], Path]:
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError(
            "TensorFlow is not installed. Install it with: pip install tensorflow"
        ) from e

    np.random.seed(int(random_state))
    try:
        tf.random.set_seed(int(random_state))
    except Exception:
        pass

    hourly_csv = processed_dir / "global_active_power_hourly.csv"
    train_csv = processed_dir / "global_active_power_train.csv"
    val_csv = processed_dir / "global_active_power_val.csv"
    test_csv = processed_dir / "global_active_power_test.csv"

    if not hourly_csv.exists() or not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            "Missing processed CSVs. Run prepare_time_series_data.py first to generate hourly/train/val/test files."
        )

    split_bounds = _load_split_bounds(processed_dir)
    if not {"train", "val", "test"}.issubset(split_bounds):
        raise RuntimeError("Could not infer train/val/test bounds from processed split CSVs.")

    df_hourly = pd.read_csv(hourly_csv, parse_dates=["datetime"]).set_index("datetime").sort_index()
    df_hourly[target_col] = pd.to_numeric(df_hourly[target_col], errors="coerce")
    df_hourly = df_hourly.dropna(subset=[target_col])
    df_hourly.index = pd.DatetimeIndex(df_hourly.index)

    step = _infer_step(df_hourly.index)

    feature_cols = [target_col]
    if include_time_features:
        features_csv = processed_dir / "global_active_power_features.csv"
        if not features_csv.exists():
            raise FileNotFoundError("Missing global_active_power_features.csv but include_time_features=True.")
        df_feat = _load_indexed_csv(features_csv)
        keep = [
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "is_weekend",
            "is_business_hour",
            "day_of_week",
        ]
        for c in keep:
            if c not in df_feat.columns:
                raise ValueError(f"Expected feature column '{c}' not found in global_active_power_features.csv.")
        df = df_feat[[target_col] + keep].copy()
        df = df.loc[df_hourly.index.min() : df_hourly.index.max()]
    else:
        df = df_hourly[[target_col]].copy()

    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna()

    train_start, train_end = split_bounds["train"]
    train_mask = (df.index >= train_start) & (df.index <= train_end)
    if not np.any(train_mask):
        raise RuntimeError("Training split bounds do not overlap the hourly dataset used for LSTM.")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_fit = df.loc[train_mask, feature_cols].to_numpy(dtype="float32")
    y_train_fit = df.loc[train_mask, [target_col]].to_numpy(dtype="float32")
    scaler_X.fit(X_train_fit)
    scaler_y.fit(y_train_fit)

    X_all = scaler_X.transform(df[feature_cols].to_numpy(dtype="float32")).astype("float32")
    y_all = scaler_y.transform(df[[target_col]].to_numpy(dtype="float32")).reshape(-1).astype("float32")

    ds_all = _build_sequences(
        features=X_all,
        target=y_all,
        timestamps=pd.DatetimeIndex(df.index),
        lookback=int(lookback_hours),
        step=step,
        horizon_hours=int(horizon_hours),
    )

    ds_train = _subset_sequences(ds_all, *split_bounds["train"])
    ds_val = _subset_sequences(ds_all, *split_bounds["val"])
    ds_test = _subset_sequences(ds_all, *split_bounds["test"])

    n_features = int(ds_train.X.shape[-1])

    inputs = tf.keras.layers.Input(shape=(int(lookback_hours), n_features))
    x = inputs
    for i, units in enumerate(lstm_units):
        return_sequences = i < (len(lstm_units) - 1)
        x = tf.keras.layers.LSTM(int(units), return_sequences=return_sequences)(x)
        if float(dropout) > 0.0:
            x = tf.keras.layers.Dropout(float(dropout))(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss="mse",
    )

    out_dir = outputs_dir / "lstm"
    if int(horizon_hours) != 1:
        out_dir = out_dir / f"h{int(horizon_hours)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / f"best_model_h{int(horizon_hours)}.keras"
    manifests_dir = Path(outputs_dir) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(patience),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    train_start = time.perf_counter()
    history = model.fit(
        ds_train.X,
        ds_train.y,
        validation_data=(ds_val.X, ds_val.y),
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=callbacks,
        verbose=1,
    )
    train_seconds = float(time.perf_counter() - train_start)

    history_path = out_dir / "training_history.png"
    _plot_history(history.history, history_path)
    (out_dir / "training_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    infer_start = time.perf_counter()
    y_pred_scaled = model.predict(ds_test.X, batch_size=int(batch_size), verbose=0).reshape(-1, 1).astype("float32")
    infer_seconds = float(time.perf_counter() - infer_start)
    y_true = scaler_y.inverse_transform(ds_test.y.reshape(-1, 1)).reshape(-1).astype("float64")
    y_pred = scaler_y.inverse_transform(y_pred_scaled).reshape(-1).astype("float64")

    pred_df = pd.DataFrame({"datetime": ds_test.y_index, "y_true": y_true, "y_pred": y_pred})
    pred_path = out_dir / f"lstm_h{int(horizon_hours)}_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    plot_path = out_dir / f"lstm_h{int(horizon_hours)}_pred_vs_actual.png"
    _plot_pred_vs_actual(
        y_true=y_true,
        y_pred=y_pred,
        index=ds_test.y_index,
        out_path=plot_path,
        title=f"LSTM predictions vs actuals (test, horizon={int(horizon_hours)}h)",
    )

    lstm_metrics = evaluate_forecast(y_true=y_true, y_pred=y_pred)
    metrics_row = {
        "Model": "LSTM",
        "MAE": lstm_metrics["MAE"],
        "RMSE": lstm_metrics["RMSE"],
        "MAPE": lstm_metrics["MAPE_%"],
        "Train_s": train_seconds,
        "Infer_s": infer_seconds,
        "Horizon_h": int(horizon_hours),
    }
    metrics_path = out_dir / f"lstm_h{int(horizon_hours)}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_row, indent=2, sort_keys=True), encoding="utf-8")

    others = _try_load_other_metrics(processed_dir=processed_dir, outputs_dir=outputs_dir)
    comparison = pd.concat([others, pd.DataFrame([metrics_row])], ignore_index=True)
    if not comparison.empty and "RMSE" in comparison.columns:
        comparison = comparison.sort_values(by="RMSE", ascending=True, kind="mergesort")
    comparison_path = out_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    scaler_x_path = None
    scaler_y_path = None
    try:
        import joblib

        scaler_x_path = out_dir / f"scaler_X_h{int(horizon_hours)}.joblib"
        scaler_y_path = out_dir / f"scaler_y_h{int(horizon_hours)}.joblib"
        joblib.dump(scaler_X, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
    except Exception:
        scaler_x_path = None
        scaler_y_path = None

    manifest = {
        "model_name": "LSTM",
        "model_family": "LSTM",
        "horizon_hours": int(horizon_hours),
        "split": "test",
        "metrics": {"MAE": lstm_metrics["MAE"], "RMSE": lstm_metrics["RMSE"], "MAPE": lstm_metrics["MAPE_%"]},
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
        "predictions_csv": to_relpath(pred_path),
        "model_path": to_relpath(best_model_path),
        "extra": {
            "lookback_hours": int(lookback_hours),
            "include_time_features": bool(include_time_features),
            "lstm_units": [int(u) for u in lstm_units],
            "dropout": float(dropout),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "patience": int(patience),
            "learning_rate": float(learning_rate),
            "scaler_X_path": to_relpath(scaler_x_path) if scaler_x_path is not None else None,
            "scaler_y_path": to_relpath(scaler_y_path) if scaler_y_path is not None else None,
        },
    }
    write_json(manifests_dir / f"lstm_{slugify('lstm')}_h{int(horizon_hours)}.json", manifest)

    return metrics_row, out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--target-col", type=str, default="Global_active_power")
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--horizon-hours", type=int, default=1)
    parser.add_argument("--lstm-units", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--include-time-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    metrics_row, out_dir = run_lstm(
        processed_dir=Path(args.processed_dir),
        outputs_dir=Path(args.outputs_dir),
        target_col=str(args.target_col),
        lookback_hours=int(args.lookback_hours),
        horizon_hours=int(args.horizon_hours),
        lstm_units=[int(u) for u in args.lstm_units],
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        learning_rate=float(args.learning_rate),
        include_time_features=bool(args.include_time_features),
        random_state=int(args.random_state),
    )

    with pd.option_context("display.max_columns", 20, "display.width", 140):
        print("LSTM test metrics:")
        print(pd.DataFrame([metrics_row]).to_string(index=False))
        print()
        print("Saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
