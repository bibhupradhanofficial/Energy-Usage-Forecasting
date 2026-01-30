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

from forecasting_baselines import evaluate_forecast
from forecasting_baselines import infer_step
from model_artifacts import slugify, to_relpath, write_json


@dataclass(frozen=True)
class ModelRun:
    name: str
    y_pred: pd.Series
    metrics: dict[str, float]
    best_params: dict[str, object] | None
    best_cv_mae: float | None
    feature_importance: pd.Series | None
    fit_seconds: float
    infer_seconds: float
    model_path: Path | None


def load_feature_split(csv_path: Path, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in {csv_path.name}.")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    return df


def get_feature_target_matrices(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        raise ValueError("No feature columns found.")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    if np.any(~np.isfinite(X)):
        raise ValueError("Found non-finite values in features. Re-run feature generation with dropna=True.")
    if np.any(~np.isfinite(y)):
        raise ValueError("Found non-finite values in target.")
    return X, y, feature_cols


def plot_predictions_vs_actual(y_test: pd.Series, y_pred: pd.Series, title: str, out_path: Path) -> None:
    y_test, y_pred = y_test.align(y_pred, join="inner")
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.index, y_test.to_numpy(), label="Actual", linewidth=1.1)
    plt.plot(y_pred.index, y_pred.to_numpy(), label="Predicted", linewidth=1.1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(y_test.name or "value")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_feature_importance(importance: pd.Series, title: str, out_path: Path, top_n: int = 20) -> None:
    imp = importance.sort_values(ascending=False).head(int(top_n)).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(imp.index.astype(str), imp.to_numpy())
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _try_get_feature_importance(model: object, feature_cols: list[str]) -> pd.Series | None:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None
    imp_arr = np.asarray(imp, dtype=float)
    if imp_arr.ndim != 1 or len(imp_arr) != len(feature_cols):
        return None
    return pd.Series(imp_arr, index=feature_cols, name="importance")


def run_ml_models(
    processed_dir: Path,
    target_col: str,
    include_val_in_train: bool,
    cv_splits: int,
    n_iter: int,
    random_state: int,
    enable_lightgbm: str,
    large_data_threshold: int,
    horizon_hours: int,
    outputs_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    train_csv = processed_dir / "global_active_power_train.csv"
    val_csv = processed_dir / "global_active_power_val.csv"
    test_csv = processed_dir / "global_active_power_test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Missing processed CSVs. Run prepare_time_series_data.py first.")

    df_train = load_feature_split(train_csv, target_col=target_col)
    df_test = load_feature_split(test_csv, target_col=target_col)

    if include_val_in_train and val_csv.exists():
        df_val = load_feature_split(val_csv, target_col=target_col)
        df_fit = pd.concat([df_train, df_val]).sort_index()
    else:
        df_fit = df_train

    step = infer_step(pd.DatetimeIndex(df_fit.index))
    horizon_hours = int(horizon_hours)
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be >= 1.")
    horizon_offset = step * horizon_hours

    y_fit_shift = df_fit[target_col].shift(periods=-horizon_hours)
    fit_mask = y_fit_shift.notna()
    df_fit_h = df_fit.loc[fit_mask].copy()
    y_fit_h = pd.to_numeric(y_fit_shift.loc[fit_mask], errors="coerce").to_numpy(dtype=float)

    y_test_shift = df_test[target_col].shift(periods=-horizon_hours)
    test_mask = y_test_shift.notna()
    df_test_h = df_test.loc[test_mask].copy()
    y_test_arr = pd.to_numeric(y_test_shift.loc[test_mask], errors="coerce").to_numpy(dtype=float)
    target_index = pd.DatetimeIndex(df_test_h.index) + horizon_offset
    y_test = pd.Series(y_test_arr, index=target_index, name=target_col)

    X_fit, _, feature_cols = get_feature_target_matrices(df_fit_h, target_col=target_col)
    X_test, _, _ = get_feature_target_matrices(df_test_h, target_col=target_col)

    out_dir = Path(outputs_dir) / "ml"
    if horizon_hours != 1:
        out_dir = out_dir / f"h{horizon_hours}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = Path(outputs_dir) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    model_runs: list[ModelRun] = []

    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as e:
        raise RuntimeError("scikit-learn is required for ML models. Install scikit-learn and re-run.") from e

    rf = RandomForestRegressor(random_state=int(random_state), n_jobs=-1)
    rf_space = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 6, 10, 16],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.6, 0.8],
    }

    rf_run = _tune_and_fit_with_importance(
        name="RandomForest",
        estimator=rf,
        param_space=rf_space,
        X_fit=X_fit,
        y_fit=y_fit_h,
        X_test=X_test,
        test_index=target_index,
        y_test=y_test,
        feature_cols=feature_cols,
        cv_splits=cv_splits,
        n_iter=n_iter,
        random_state=random_state,
        out_dir=out_dir,
        manifests_dir=manifests_dir,
        horizon_hours=horizon_hours,
    )
    model_runs.append(rf_run)

    xgb_run = _maybe_run_xgboost(
        X_fit=X_fit,
        y_fit=y_fit_h,
        X_test=X_test,
        test_index=target_index,
        y_test=y_test,
        feature_cols=feature_cols,
        cv_splits=cv_splits,
        n_iter=n_iter,
        random_state=random_state,
        out_dir=out_dir,
        manifests_dir=manifests_dir,
        horizon_hours=horizon_hours,
    )
    if xgb_run is not None:
        model_runs.append(xgb_run)
    else:
        print("XGBoost not available (xgboost not installed). Skipping XGBoost model.")

    should_try_lgbm = enable_lightgbm.lower() in {"1", "true", "yes", "y", "on"}
    if enable_lightgbm.lower() == "auto":
        should_try_lgbm = len(df_fit) >= int(large_data_threshold)

    lgbm_run = None
    if should_try_lgbm:
        lgbm_run = _maybe_run_lightgbm(
            X_fit=X_fit,
            y_fit=y_fit_h,
            X_test=X_test,
            test_index=target_index,
            y_test=y_test,
            feature_cols=feature_cols,
            cv_splits=cv_splits,
            n_iter=n_iter,
            random_state=random_state,
            out_dir=out_dir,
            manifests_dir=manifests_dir,
            horizon_hours=horizon_hours,
        )
        if lgbm_run is not None:
            model_runs.append(lgbm_run)
        else:
            print("LightGBM not available (lightgbm not installed). Skipping LightGBM model.")

    rows = []
    for r in model_runs:
        rows.append(
            {
                "Model": r.name,
                "MAE": r.metrics["MAE"],
                "RMSE": r.metrics["RMSE"],
                "MAPE": r.metrics["MAPE"],
                "BestCV_MAE": r.best_cv_mae,
                "Train_s": r.fit_seconds,
                "Infer_s": r.infer_seconds,
            }
        )
    comparison = pd.DataFrame(rows).sort_values(by="RMSE", ascending=True, kind="mergesort")
    comparison_path = out_dir / "ml_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    best_model_name = str(comparison.iloc[0]["Model"]) if not comparison.empty else None
    best_importance_path = out_dir / "feature_importance_best.png"
    if best_model_name is not None:
        for r in model_runs:
            pred_path = out_dir / f"{slugify(r.name)}_h{horizon_hours}_test_predictions.csv"
            pd.DataFrame(
                {"datetime": r.y_pred.index, "y_true": y_test.to_numpy(), "y_pred": r.y_pred.to_numpy()}
            ).to_csv(pred_path, index=False)

            if r.best_params is not None:
                params_path = out_dir / f"{r.name.lower()}_best_params.json"
                params_path.write_text(json.dumps(r.best_params, indent=2, sort_keys=True), encoding="utf-8")

            plot_predictions_vs_actual(
                y_test=y_test,
                y_pred=r.y_pred,
                title=f"{r.name} predictions vs actuals (test, horizon={horizon_hours}h)",
                out_path=out_dir / f"{slugify(r.name)}_h{horizon_hours}_pred_vs_actual.png",
            )

            if r.feature_importance is not None:
                plot_feature_importance(
                    r.feature_importance,
                    title=f"{r.name} feature importance (top 20)",
                    out_path=out_dir / f"{slugify(r.name)}_h{horizon_hours}_feature_importance.png",
                    top_n=20,
                )
                if r.name == best_model_name:
                    plot_feature_importance(
                        r.feature_importance,
                        title=f"Best model feature importance: {r.name}",
                        out_path=best_importance_path,
                        top_n=25,
                    )

    return comparison, out_dir


def _tune_and_fit_with_importance(
    name: str,
    estimator: object,
    param_space: dict[str, object] | None,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    test_index: pd.DatetimeIndex,
    y_test: pd.Series,
    feature_cols: list[str],
    cv_splits: int,
    n_iter: int,
    random_state: int,
    out_dir: Path,
    manifests_dir: Path,
    horizon_hours: int,
) -> ModelRun:
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    cv = TimeSeriesSplit(n_splits=int(cv_splits))

    best_params = None
    best_cv_mae = None
    best_estimator = estimator

    fit_start = time.perf_counter()
    if param_space:
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_space,
            n_iter=int(n_iter),
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            random_state=int(random_state),
            refit=True,
        )
        search.fit(X_fit, y_fit)
        best_estimator = search.best_estimator_
        best_params = dict(search.best_params_)
        best_cv_mae = float(-search.best_score_)
    else:
        best_estimator.fit(X_fit, y_fit)
    fit_seconds = float(time.perf_counter() - fit_start)

    infer_start = time.perf_counter()
    y_hat = best_estimator.predict(X_test).astype("float64")
    infer_seconds = float(time.perf_counter() - infer_start)
    y_pred = pd.Series(y_hat, index=test_index, name=f"{name}_pred")
    result = evaluate_forecast(model_name=name, y_true=y_test, y_pred=y_pred, metadata={})
    importance = _try_get_feature_importance(best_estimator, feature_cols=feature_cols)

    model_path = None
    try:
        import joblib

        model_path = out_dir / f"{slugify(name)}_h{int(horizon_hours)}.joblib"
        joblib.dump(best_estimator, model_path)
    except Exception:
        model_path = None

    manifest = {
        "model_name": name,
        "model_family": "ML",
        "horizon_hours": int(horizon_hours),
        "split": "test",
        "metrics": dict(result.metrics),
        "train_seconds": fit_seconds,
        "infer_seconds": infer_seconds,
        "predictions_csv": to_relpath(out_dir / f"{slugify(name)}_h{int(horizon_hours)}_test_predictions.csv"),
        "model_path": to_relpath(model_path) if model_path is not None else None,
        "feature_cols": list(feature_cols),
        "extra": {"best_params": best_params, "best_cv_mae": best_cv_mae},
    }
    write_json(manifests_dir / f"ml_{slugify(name)}_h{int(horizon_hours)}.json", manifest)

    return ModelRun(
        name=name,
        y_pred=result.y_pred,
        metrics=dict(result.metrics),
        best_params=best_params,
        best_cv_mae=best_cv_mae,
        feature_importance=importance,
        fit_seconds=fit_seconds,
        infer_seconds=infer_seconds,
        model_path=model_path,
    )


def _maybe_run_xgboost(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    test_index: pd.DatetimeIndex,
    y_test: pd.Series,
    feature_cols: list[str],
    cv_splits: int,
    n_iter: int,
    random_state: int,
    out_dir: Path,
    manifests_dir: Path,
    horizon_hours: int,
) -> ModelRun | None:
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=int(random_state),
        n_jobs=-1,
        tree_method="hist",
    )

    space = {
        "n_estimators": [300, 600, 1000],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    return _tune_and_fit_with_importance(
        name="XGBoost",
        estimator=xgb,
        param_space=space,
        X_fit=X_fit,
        y_fit=y_fit,
        X_test=X_test,
        test_index=test_index,
        y_test=y_test,
        feature_cols=feature_cols,
        cv_splits=cv_splits,
        n_iter=n_iter,
        random_state=random_state,
        out_dir=out_dir,
        manifests_dir=manifests_dir,
        horizon_hours=horizon_hours,
    )


def _maybe_run_lightgbm(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    test_index: pd.DatetimeIndex,
    y_test: pd.Series,
    feature_cols: list[str],
    cv_splits: int,
    n_iter: int,
    random_state: int,
    out_dir: Path,
    manifests_dir: Path,
    horizon_hours: int,
) -> ModelRun | None:
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        return None

    lgbm = LGBMRegressor(random_state=int(random_state), n_jobs=-1)
    space = {
        "n_estimators": [300, 600, 1000],
        "max_depth": [-1, 6, 10, 16],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "num_leaves": [31, 63, 127],
    }

    return _tune_and_fit_with_importance(
        name="LightGBM",
        estimator=lgbm,
        param_space=space,
        X_fit=X_fit,
        y_fit=y_fit,
        X_test=X_test,
        test_index=test_index,
        y_test=y_test,
        feature_cols=feature_cols,
        cv_splits=cv_splits,
        n_iter=n_iter,
        random_state=random_state,
        out_dir=out_dir,
        manifests_dir=manifests_dir,
        horizon_hours=horizon_hours,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--target-col", type=str, default="Global_active_power")
    parser.add_argument("--include-val-in-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--enable-lightgbm", type=str, default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--large-data-threshold", type=int, default=20000)
    parser.add_argument("--horizon-hours", type=int, default=1)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    comparison, out_dir = run_ml_models(
        processed_dir=Path(args.processed_dir),
        target_col=str(args.target_col),
        include_val_in_train=bool(args.include_val_in_train),
        cv_splits=int(args.cv_splits),
        n_iter=int(args.n_iter),
        random_state=int(args.random_state),
        enable_lightgbm=str(args.enable_lightgbm),
        large_data_threshold=int(args.large_data_threshold),
        horizon_hours=int(args.horizon_hours),
        outputs_dir=Path(args.outputs_dir),
    )

    with pd.option_context("display.max_columns", 20, "display.width", 140):
        print(comparison.round(6).to_string(index=False))
    print()
    print("Saved outputs:", out_dir)


if __name__ == "__main__":
    main()
