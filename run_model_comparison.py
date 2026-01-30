from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_artifacts import slugify
from stats_tests import diebold_mariano_test


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(x: object) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns or "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Predictions CSV missing required columns: {path}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["datetime", "y_true", "y_pred"]).sort_values("datetime")
    return df


def _peak_hours_from_train(train_csv: Path, target_col: str) -> set[int]:
    df = pd.read_csv(train_csv, index_col=0, parse_dates=True)
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in {train_csv.name}.")
    y = pd.to_numeric(df[target_col], errors="coerce").dropna()
    y.index = pd.DatetimeIndex(y.index)
    hourly_mean = y.groupby(y.index.hour).mean()
    thresh = float(np.nanpercentile(hourly_mean.to_numpy(dtype=float), 75.0))
    peak = set(int(h) for h, v in hourly_mean.items() if float(v) >= thresh)
    return peak


def _metric_bar_plot(df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["MAE", "RMSE", "MAPE"]
    plot_df = df.copy()
    plot_df["Label"] = plot_df["Model"].astype(str)
    plot_df = plot_df.sort_values(by="RMSE", ascending=True, kind="mergesort")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, m in zip(axes, metrics, strict=False):
        ax.bar(plot_df["Label"], plot_df[m].to_numpy(dtype=float))
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _top3_overlay_plot(pred_frames: list[tuple[str, pd.DataFrame]], out_path: Path, title: str) -> None:
    if not pred_frames:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    base = pred_frames[0][1].copy()
    ax.plot(base["datetime"], base["y_true"], label="Actual", linewidth=1.2)
    for name, df in pred_frames:
        ax.plot(df["datetime"], df["y_pred"], label=name, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _residual_plots(best_name: str, pred_df: pd.DataFrame, out_dir: Path) -> None:
    df = pred_df.copy()
    df["residual"] = df["y_true"] - df["y_pred"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["datetime"], df["residual"], linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Residuals over time: {best_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("y_true - y_pred")
    fig.tight_layout()
    fig.savefig(out_dir / "residuals_time.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["residual"].to_numpy(dtype=float), bins=60)
    ax.set_title(f"Residual distribution: {best_name}")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "residuals_hist.png", dpi=160)
    plt.close(fig)


def build_comparison(outputs_dir: Path, processed_dir: Path, target_col: str) -> tuple[pd.DataFrame, Path]:
    manifests_dir = outputs_dir / "manifests"
    out_dir = outputs_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = sorted(manifests_dir.glob("*.json"))
    rows: list[dict[str, object]] = []
    for mp in manifest_paths:
        m = _load_manifest(mp)
        metrics = dict(m.get("metrics", {}) or {})
        rows.append(
            {
                "Model": str(m.get("model_name", mp.stem)),
                "Family": str(m.get("model_family", "")),
                "Horizon_h": int(m.get("horizon_hours", 1)),
                "MAE": _as_float(metrics.get("MAE")),
                "RMSE": _as_float(metrics.get("RMSE")),
                "MAPE": _as_float(metrics.get("MAPE")),
                "Train_s": _as_float(m.get("train_seconds")),
                "Infer_s": _as_float(m.get("infer_seconds")),
                "Predictions": str(m.get("predictions_csv") or ""),
                "ModelPath": str(m.get("model_path") or ""),
                "Manifest": str(mp),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Horizon_h", "RMSE"], ascending=[True, True], kind="mergesort")
    df.to_csv(out_dir / "model_comparison_all.csv", index=False)

    h1 = df[df["Horizon_h"] == 1].copy()
    if not h1.empty:
        _metric_bar_plot(h1, out_dir / "metrics_bar_h1.png")

        top3 = h1.sort_values(by="RMSE", ascending=True, kind="mergesort").head(3)
        pred_frames: list[tuple[str, pd.DataFrame]] = []
        for _, r in top3.iterrows():
            p = Path(str(r["Predictions"]))
            if not p.exists():
                continue
            pred_frames.append((str(r["Model"]), _load_predictions(p)))
        if pred_frames:
            _top3_overlay_plot(pred_frames, out_dir / "top3_overlay_h1.png", title="Top-3 models vs actual (horizon=1h)")
            best_name, best_df = pred_frames[0]
            _residual_plots(best_name, best_df, out_dir)

        if pred_frames:
            best_name, best_df = pred_frames[0]
            dm_rows: list[dict[str, object]] = []
            for _, r in h1.sort_values(by="RMSE", ascending=True, kind="mergesort").iterrows():
                name = str(r["Model"])
                if name == best_name:
                    continue
                p = Path(str(r["Predictions"]))
                if not p.exists():
                    continue
                dfp = _load_predictions(p)
                merged = best_df.merge(dfp, on="datetime", suffixes=("_best", "_alt"), how="inner")
                res = diebold_mariano_test(
                    y_true=merged["y_true_best"].to_numpy(dtype=float),
                    y_pred_1=merged["y_pred_best"].to_numpy(dtype=float),
                    y_pred_2=merged["y_pred_alt"].to_numpy(dtype=float),
                    h=1,
                    loss="se",
                )
                dm_rows.append(
                    {
                        "Best": best_name,
                        "ComparedTo": name,
                        "DM_stat": res.statistic,
                        "p_value": res.p_value,
                        "n": res.n,
                    }
                )
            if dm_rows:
                pd.DataFrame(dm_rows).sort_values(by="p_value", ascending=True, kind="mergesort").to_csv(
                    out_dir / "dm_test_h1_vs_best.csv", index=False
                )

        train_csv = processed_dir / "global_active_power_train.csv"
        if train_csv.exists():
            peak_hours = _peak_hours_from_train(train_csv, target_col=target_col)
            peak_rows: list[dict[str, object]] = []
            for _, r in h1.iterrows():
                p = Path(str(r["Predictions"]))
                if not p.exists():
                    continue
                pdf = _load_predictions(p)
                hours = pd.DatetimeIndex(pdf["datetime"]).hour
                is_peak = np.isin(hours.to_numpy(dtype=int), np.asarray(sorted(peak_hours), dtype=int))
                if not np.any(is_peak) or np.all(is_peak):
                    continue
                abs_err = np.abs(pdf["y_true"].to_numpy(dtype=float) - pdf["y_pred"].to_numpy(dtype=float))
                peak_mae = float(np.mean(abs_err[is_peak]))
                off_mae = float(np.mean(abs_err[~is_peak]))
                peak_rows.append({"Model": str(r["Model"]), "PeakHours": ",".join(str(h) for h in sorted(peak_hours)), "MAE_peak": peak_mae, "MAE_offpeak": off_mae})
            if peak_rows:
                pd.DataFrame(peak_rows).sort_values(by="MAE_peak", ascending=True, kind="mergesort").to_csv(
                    out_dir / "peak_offpeak_h1.csv", index=False
                )

    horizon_best_rows: list[dict[str, object]] = []
    for h in sorted(df["Horizon_h"].unique().tolist()) if not df.empty else []:
        sub = df[df["Horizon_h"] == h].sort_values(by="RMSE", ascending=True, kind="mergesort")
        if sub.empty:
            continue
        best = sub.iloc[0]
        horizon_best_rows.append({"Horizon_h": int(h), "BestModel": str(best["Model"]), "RMSE": float(best["RMSE"]), "MAE": float(best["MAE"]), "MAPE": float(best["MAPE"])})
    if horizon_best_rows:
        pd.DataFrame(horizon_best_rows).to_csv(out_dir / "best_by_horizon.csv", index=False)

    return df, out_dir


def select_and_stage_best_model(df: pd.DataFrame, outputs_dir: Path, preferred_horizon: int) -> Path | None:
    if df.empty:
        return None
    preferred_horizon = int(preferred_horizon)
    candidates = df[df["Horizon_h"] == preferred_horizon].copy()
    if candidates.empty:
        candidates = df[df["Horizon_h"] == 1].copy()
    if candidates.empty:
        return None
    best = candidates.sort_values(by="RMSE", ascending=True, kind="mergesort").iloc[0]
    manifest_path = Path(str(best["Manifest"]))
    manifest = _load_manifest(manifest_path) if manifest_path.exists() else {}
    family = str(best["Family"])
    model_path = Path(str(best["ModelPath"])) if str(best["ModelPath"]) else None
    if model_path is None or not model_path.exists():
        return None

    deploy_dir = outputs_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    if family.upper() == "LSTM":
        try:
            import tensorflow as tf

            export_dir = deploy_dir / f"best_lstm_savedmodel_h{int(best['Horizon_h'])}"
            if export_dir.exists():
                shutil.rmtree(export_dir)
            model = tf.keras.models.load_model(model_path)
            try:
                model.export(str(export_dir))
            except Exception:
                model.save(str(export_dir))
            staged = export_dir
        except Exception:
            suffix = model_path.suffix.lower()
            staged = deploy_dir / f"best_model_h{int(best['Horizon_h'])}{suffix}"
            shutil.copy2(model_path, staged)
    else:
        suffix = model_path.suffix.lower()
        staged = deploy_dir / f"best_model_h{int(best['Horizon_h'])}{suffix}"
        shutil.copy2(model_path, staged)

    extra = dict(manifest.get("extra", {}) or {})
    for k in ("scaler_X_path", "scaler_y_path"):
        p = extra.get(k)
        if not p:
            continue
        src = Path(str(p))
        if src.exists():
            shutil.copy2(src, deploy_dir / Path(str(p)).name)

    meta = {
        "selected_model": str(best["Model"]),
        "model_family": family,
        "horizon_hours": int(best["Horizon_h"]),
        "metrics": {"MAE": float(best["MAE"]), "RMSE": float(best["RMSE"]), "MAPE": float(best["MAPE"])},
        "source_model_path": str(model_path).replace("\\", "/"),
        "staged_path": str(staged).replace("\\", "/") if staged is not None else None,
    }
    (deploy_dir / "best_model_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return staged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--target-col", type=str, default="Global_active_power")
    parser.add_argument("--preferred-horizon", type=int, default=24)
    parser.add_argument("--no-stage-best", action="store_true")
    args = parser.parse_args()

    df, out_dir = build_comparison(outputs_dir=Path(args.outputs_dir), processed_dir=Path(args.processed_dir), target_col=str(args.target_col))
    staged = None
    if not args.no_stage_best:
        staged = select_and_stage_best_model(df, outputs_dir=Path(args.outputs_dir), preferred_horizon=int(args.preferred_horizon))
    print("Saved comparison outputs to:", out_dir)
    if staged is not None:
        print("Staged best model artifact:", staged)


if __name__ == "__main__":
    main()

