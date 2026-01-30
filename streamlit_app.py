from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

BASELINE_VARIANTS = [
    "Seasonal naive (daily)",
    "Seasonal naive (weekly)",
    "Naive (last value)",
    "Moving average (24)",
    "Moving average (48)",
]

FORECAST_MODES = ["Multi-step (recursive, 1h model)", "Direct (single point at horizon)"]


@dataclass(frozen=True)
class SidebarInputs:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    compare_mode: bool
    selected_models: tuple[str, ...]
    seasonal_naive_variant: str
    moving_average_variant: str
    baseline_reference: str
    horizon_hours: int
    forecast_mode: str
    realtime_mode: bool
    refresh_seconds: int
    stream_speed_hours: int
    auto_generate: bool
    generate_clicked: bool


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    canonical_family: str | None
    canonical_model: str | None
    artifact_model_path: Path | None
    manifest_path: Path | None
    horizon_hours: int | None
    horizon_to_model_path: dict[int, Path] | None = None
    horizon_to_manifest_path: dict[int, Path] | None = None


def _safe_secrets_get(key: str) -> str | None:
    try:
        value = st.secrets.get(key)  # type: ignore[call-arg]
        if value is None:
            return None
        return str(value)
    except Exception:
        return None


def _debug_enabled() -> bool:
    raw = os.getenv("ENERGY_APP_DEBUG") or _safe_secrets_get("ENERGY_APP_DEBUG")
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _path_from_env_or_default(key: str, default: Path) -> Path:
    raw = os.getenv(key) or _safe_secrets_get(key)
    if not raw:
        return default
    try:
        return Path(raw).expanduser().resolve()
    except Exception:
        return default


def _import_optional(module_name: str) -> object | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_joblib_model(path: Path) -> object:
    joblib = _import_optional("joblib")
    if joblib is None:
        raise RuntimeError("joblib is required to load this model. Install scikit-learn (includes joblib).")
    return joblib.load(path)  # type: ignore[attr-defined]


@st.cache_resource(show_spinner=False)
def load_sarimax_results(path: Path) -> object:
    sm = _import_optional("statsmodels")
    if sm is None:
        raise RuntimeError("statsmodels is required to load SARIMAX models.")
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults

    return SARIMAXResults.load(path)


@st.cache_resource(show_spinner=False)
def load_prophet_model(path: Path) -> object:
    if _import_optional("prophet") is None:
        raise RuntimeError("prophet is not installed. Install it with: pip install prophet")
    return load_joblib_model(path)


def _supports_run_every_fragment() -> bool:
    frag = getattr(st, "fragment", None)
    if frag is None:
        return False
    try:
        sig = inspect.signature(frag)
        return "run_every" in sig.parameters
    except Exception:
        return False


def _width_kwargs(widget: object, use_container_width: bool) -> dict[str, object]:
    try:
        sig = inspect.signature(widget)
    except Exception:
        return {"use_container_width": use_container_width}
    if "width" in sig.parameters:
        return {"width": "stretch" if use_container_width else "content"}
    if "use_container_width" in sig.parameters:
        return {"use_container_width": use_container_width}
    return {}


def _maybe_autorefresh(enabled: bool, seconds: int) -> None:
    if not enabled:
        return
    if seconds < 1:
        return
    auto = getattr(st, "autorefresh", None)
    if callable(auto):
        auto(interval=int(seconds) * 1000, key="_autorefresh")
        return
    if _supports_run_every_fragment():
        @st.fragment(run_every=seconds)  # type: ignore[misc]
        def _tick() -> None:
            st.session_state["_autorefresh_tick"] = time.time()

        _tick()
    else:
        st.info("Auto-refresh is not supported by this Streamlit version. Use the browser refresh button.")


@st.cache_data(show_spinner=False)
def load_hourly_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"datetime", "Global_active_power"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Expected columns {sorted(expected_cols)} in {csv_path.name}, got {sorted(df.columns)}."
        )
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    df = df.dropna(subset=["datetime", "Global_active_power"]).sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime"], keep="first").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_model_comparison(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"Model", "Family", "Horizon_h", "MAE", "RMSE", "MAPE", "Predictions"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Expected columns {sorted(expected_cols)} in {csv_path.name}, got {sorted(df.columns)}."
        )
    df["Horizon_h"] = pd.to_numeric(df["Horizon_h"], errors="coerce")
    df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")
    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
    df["MAPE"] = pd.to_numeric(df["MAPE"], errors="coerce")
    df["Predictions"] = df["Predictions"].astype(str)
    df = df.dropna(subset=["Model", "Family", "Horizon_h", "MAE", "RMSE", "MAPE", "Predictions"])
    df["Horizon_h"] = df["Horizon_h"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    return df


def filter_date_range(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    return df.loc[mask].copy()


def compute_key_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"latest_kw": np.nan, "avg_daily_kwh": np.nan, "peak_kw": np.nan, "last_24h_kwh": np.nan}

    s = df.set_index("datetime")["Global_active_power"].sort_index()
    latest_kw = float(s.iloc[-1])
    peak_kw = float(s.max())

    daily_kwh = s.resample("D").sum(min_count=1)
    avg_daily_kwh = float(daily_kwh.mean()) if len(daily_kwh) else np.nan

    last_24h = s.iloc[-24:] if len(s) >= 24 else s
    last_24h_kwh = float(last_24h.sum()) if len(last_24h) else np.nan

    return {
        "latest_kw": latest_kw,
        "avg_daily_kwh": avg_daily_kwh,
        "peak_kw": peak_kw,
        "last_24h_kwh": last_24h_kwh,
    }


def generate_forecast_placeholder(history: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["datetime", "yhat"])

    s = history.set_index("datetime")["Global_active_power"].sort_index()
    step = pd.Timedelta(hours=1)
    last_ts = pd.Timestamp(s.index[-1])
    future_index = pd.date_range(start=last_ts + step, periods=int(horizon_hours), freq=step)
    yhat = np.full(shape=len(future_index), fill_value=float(s.iloc[-1]), dtype=float)
    return pd.DataFrame({"datetime": future_index, "yhat": yhat})


def compute_improvement_pct(baseline_value: float, model_value: float) -> float:
    if not np.isfinite(baseline_value) or baseline_value == 0:
        return np.nan
    if not np.isfinite(model_value):
        return np.nan
    return float((baseline_value - model_value) / baseline_value * 100.0)


def format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if not np.any(mask):
        return {"mae": np.nan, "rmse": np.nan, "mape_pct": np.nan}

    err = y_true_arr[mask] - y_pred_arr[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    denom_mask = mask & (np.asarray(y_true_arr, dtype=float) != 0.0)
    if np.any(denom_mask):
        mape_pct = float(
            np.mean(np.abs((y_true_arr[denom_mask] - y_pred_arr[denom_mask]) / y_true_arr[denom_mask])) * 100.0
        )
    else:
        mape_pct = np.nan

    return {"mae": mae, "rmse": rmse, "mape_pct": mape_pct}


def make_hourly_series(df_all: pd.DataFrame) -> pd.Series:
    s = df_all.set_index("datetime")["Global_active_power"].sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.asfreq("h")


def baseline_predictions_for_timestamps(
    s_hourly: pd.Series, timestamps: pd.Series, baseline_name: str, horizon_hours: int
) -> pd.Series:
    idx = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    if baseline_name == "Seasonal naive (daily)":
        lag = 24
        pred = s_hourly.shift(lag).reindex(idx)
    elif baseline_name == "Seasonal naive (weekly)":
        lag = 168
        pred = s_hourly.shift(lag).reindex(idx)
    elif baseline_name == "Naive (last value)":
        lag = int(horizon_hours)
        pred = s_hourly.shift(lag).reindex(idx)
    elif baseline_name == "Moving average (24)":
        base = s_hourly.shift(int(horizon_hours)).rolling(window=24, min_periods=1).mean()
        pred = base.reindex(idx)
    elif baseline_name == "Moving average (48)":
        base = s_hourly.shift(int(horizon_hours)).rolling(window=48, min_periods=1).mean()
        pred = base.reindex(idx)
    else:
        pred = pd.Series(index=idx, dtype=float)
    pred.index = idx
    return pred


def build_baseline_prediction_df(
    df_all: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, baseline_name: str, horizon_hours: int
) -> pd.DataFrame:
    s_hourly = make_hourly_series(df_all)
    idx = s_hourly.loc[start:end].index
    y_true = s_hourly.reindex(idx)
    y_pred = baseline_predictions_for_timestamps(s_hourly, idx.to_series(), baseline_name, horizon_hours)
    df_pred = pd.DataFrame({"datetime": idx, "y_true": y_true.values, "y_pred": y_pred.values})
    return df_pred.dropna(subset=["datetime", "y_true", "y_pred"]).sort_values("datetime")


@st.cache_data(show_spinner=False)
def build_seasonal_heatmap(df_hist: pd.DataFrame) -> pd.DataFrame:
    if df_hist.empty:
        return pd.DataFrame()
    df = df_hist[["datetime", "Global_active_power"]].copy()
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    pivot = (
        df.pivot_table(index="hour", columns="dow", values="Global_active_power", aggfunc="mean")
        .sort_index()
        .reindex(columns=list(range(7)))
    )
    pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return pivot


def resolve_baseline_row(model_comparison: pd.DataFrame, horizon_hours: int) -> pd.Series | None:
    if model_comparison.empty:
        return None
    candidates = model_comparison[
        (model_comparison["Family"] == "Baseline") & (model_comparison["Horizon_h"] == int(horizon_hours))
    ].copy()
    if candidates.empty:
        return None
    candidates = candidates.sort_values(["RMSE", "MAE"], ascending=[True, True])
    return candidates.iloc[0]


def resolve_baseline_reference_row(
    model_comparison: pd.DataFrame, horizon_hours: int, baseline_reference: str
) -> pd.Series | None:
    if model_comparison.empty:
        return None
    if baseline_reference == "Auto (best available)":
        return resolve_baseline_row(model_comparison, horizon_hours)
    candidates = model_comparison[
        (model_comparison["Family"] == "Baseline")
        & (model_comparison["Model"] == baseline_reference)
        & (model_comparison["Horizon_h"] == int(horizon_hours))
    ]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def load_ci_forecast_if_available(root: Path, inputs: SidebarInputs) -> pd.DataFrame | None:
    model_key = inputs.selected_models[0] if inputs.selected_models else ""
    if model_key == "prophet":
        path = root / "outputs" / "prophet" / "prophet_forecast.csv"
    elif model_key == "arima":
        path = root / "outputs" / "arima" / "arima_forecast.csv"
    else:
        return None

    if not path.exists():
        return None

    df = load_predictions(path)
    expected_cols = {"ds", "yhat", "yhat_lower", "yhat_upper"}
    if not expected_cols.issubset(set(df.columns)):
        return None
    if "y_actual" in df.columns:
        df["y_actual"] = pd.to_numeric(df["y_actual"], errors="coerce")
    df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")
    df["yhat_lower"] = pd.to_numeric(df["yhat_lower"], errors="coerce")
    df["yhat_upper"] = pd.to_numeric(df["yhat_upper"], errors="coerce")
    return df.dropna(subset=["ds", "yhat", "yhat_lower", "yhat_upper"])


def render_sidebar(df_all: pd.DataFrame) -> SidebarInputs:
    st.sidebar.header("Controls")

    min_dt = pd.Timestamp(df_all["datetime"].min()).date()
    max_dt = pd.Timestamp(df_all["datetime"].max()).date()
    default_start = max(min_dt, (pd.Timestamp(max_dt) - pd.Timedelta(days=30)).date())
    default_end = max_dt

    start_date, end_date = st.sidebar.date_input(
        "Historical date range",
        value=(default_start, default_end),
        min_value=min_dt,
        max_value=max_dt,
    )

    compare_mode = st.sidebar.toggle("Comparison mode", value=False)

    baseline_reference = st.sidebar.selectbox(
        "Baseline for comparison",
        options=["Auto (best available)", *BASELINE_VARIANTS],
        index=0,
    )

    seasonal_naive_variant = "Seasonal naive (daily)"
    moving_average_variant = "Moving average (24)"
    registry = build_model_registry(get_project_root())
    model_keys = list(registry.keys())
    default_idx = 0 if model_keys else 0

    if compare_mode:
        selected = st.sidebar.multiselect(
            "Models to compare",
            options=model_keys,
            default=[model_keys[default_idx]] if model_keys else [],
            format_func=lambda k: registry[k].label if k in registry else str(k),
        )
        selected_models = tuple(selected)
    else:
        model_key = st.sidebar.selectbox(
            "Model",
            options=model_keys,
            index=default_idx,
            format_func=lambda k: registry[k].label if k in registry else str(k),
        )
        selected_models = (str(model_key),)
        spec = registry.get(str(model_key))
        if spec is not None:
            if spec.horizon_to_model_path is not None:
                horizons = ", ".join(str(h) for h in sorted(spec.horizon_to_model_path.keys()))
                st.sidebar.caption(f"Auto model supports horizons: {horizons}h. Recursive mode uses a 1h model.")
            elif spec.horizon_hours is not None and int(spec.horizon_hours) != 1:
                st.sidebar.caption(
                    f"This model was trained for h={int(spec.horizon_hours)}h. Direct mode requires selecting that horizon."
                )

    if any(k == "baseline_seasonal_naive" for k in selected_models):
        seasonal_naive_variant = st.sidebar.selectbox(
            "Seasonal naive variant",
            options=BASELINE_VARIANTS,
            index=0,
        )

    if any(k == "baseline_moving_average" for k in selected_models):
        moving_average_variant = st.sidebar.selectbox(
            "Moving average window",
            options=["Moving average (24)", "Moving average (48)"],
            index=0,
        )

    horizon_hours = st.sidebar.slider("Forecast horizon (hours)", min_value=1, max_value=168, value=24, step=1)

    forecast_mode = st.sidebar.selectbox("Forecast mode", options=FORECAST_MODES, index=0)

    realtime_mode = st.sidebar.toggle("Simulate real-time updates (optional)", value=False)
    refresh_seconds = int(
        st.sidebar.number_input("Auto-refresh interval (seconds)", min_value=0, max_value=600, value=0, step=5)
    )
    stream_speed_hours = int(
        st.sidebar.number_input("Stream step (hours per refresh)", min_value=1, max_value=24, value=1, step=1)
    )

    auto_generate = st.sidebar.toggle("Auto-generate on refresh/change", value=False)

    generate_clicked = st.sidebar.button("Generate Forecast", type="primary", **_width_kwargs(st.sidebar.button, True))

    return SidebarInputs(
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        compare_mode=bool(compare_mode),
        selected_models=tuple(selected_models),
        seasonal_naive_variant=str(seasonal_naive_variant),
        moving_average_variant=str(moving_average_variant),
        baseline_reference=str(baseline_reference),
        horizon_hours=int(horizon_hours),
        forecast_mode=str(forecast_mode),
        realtime_mode=bool(realtime_mode),
        refresh_seconds=int(refresh_seconds),
        stream_speed_hours=int(stream_speed_hours),
        auto_generate=bool(auto_generate),
        generate_clicked=bool(generate_clicked),
    )


@st.cache_data(show_spinner=False)
def build_model_registry(root: Path) -> dict[str, ModelSpec]:
    registry: dict[str, ModelSpec] = {}

    deploy_meta_path = root / "outputs" / "deploy" / "best_model_metadata.json"
    deploy_model_path = root / "outputs" / "deploy" / "best_model_h24.joblib"
    if deploy_meta_path.exists() and deploy_model_path.exists():
        meta = load_json(deploy_meta_path)
        h = meta.get("horizon_hours")
        horizon = int(h) if isinstance(h, (int, float, str)) and str(h).isdigit() else None
        fam = str(meta.get("model_family") or "ML")
        name = str(meta.get("model_name") or "Deployed")
        registry["deployed_best"] = ModelSpec(
            key="deployed_best",
            label=f"Deployed best model{f' (h={horizon}h)' if horizon else ''}",
            family=fam,
            canonical_family=fam,
            canonical_model=name,
            artifact_model_path=deploy_model_path,
            manifest_path=None,
            horizon_hours=horizon,
        )

    comparison_path = root / "outputs" / "comparison" / "model_comparison_all.csv"
    if comparison_path.exists():
        try:
            df = load_model_comparison(comparison_path)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            grouped: dict[tuple[str, str], dict[int, tuple[Path, Path | None]]] = {}
            for _, row in df.iterrows():
                family = str(row["Family"])
                model = str(row["Model"])
                h = int(row["Horizon_h"])
                model_path_raw = str(row.get("ModelPath") or "").strip()
                manifest_raw = str(row.get("Manifest") or "").strip()
                model_path = (root / model_path_raw) if model_path_raw else None
                manifest_path = (root / manifest_raw) if manifest_raw else None
                if model_path is not None and not model_path.exists():
                    model_path = None
                if manifest_path is not None and not manifest_path.exists():
                    manifest_path = None

                key = f"{family.lower()}_{model.lower()}_h{h}"
                label = f"{family}: {model} (h={h}h)"
                if key not in registry:
                    registry[key] = ModelSpec(
                        key=key,
                        label=label,
                        family=family,
                        canonical_family=family,
                        canonical_model=model,
                        artifact_model_path=model_path,
                        manifest_path=manifest_path,
                        horizon_hours=h,
                    )

                if model_path is not None:
                    grouped.setdefault((family, model), {})[int(h)] = (model_path, manifest_path)

            for (family, model), by_h in grouped.items():
                if len(by_h) < 2:
                    continue
                horizons = sorted(by_h.keys())
                key = f"{family.lower()}_{model.lower()}_auto"
                label = f"{family}: {model} (auto-match horizon: {', '.join(str(x) for x in horizons)}h)"
                if key in registry:
                    continue
                registry[key] = ModelSpec(
                    key=key,
                    label=label,
                    family=family,
                    canonical_family=family,
                    canonical_model=model,
                    artifact_model_path=None,
                    manifest_path=None,
                    horizon_hours=None,
                    horizon_to_model_path={h: p for h, (p, _) in by_h.items()},
                    horizon_to_manifest_path={h: m for h, (_, m) in by_h.items() if m is not None},
                )

    registry["baseline_seasonal_naive"] = ModelSpec(
        key="baseline_seasonal_naive",
        label="Baseline: Seasonal naive",
        family="Baseline",
        canonical_family="Baseline",
        canonical_model="Seasonal naive",
        artifact_model_path=None,
        manifest_path=None,
        horizon_hours=None,
    )

    registry["baseline_moving_average"] = ModelSpec(
        key="baseline_moving_average",
        label="Baseline: Moving average",
        family="Baseline",
        canonical_family="Baseline",
        canonical_model="Moving average",
        artifact_model_path=None,
        manifest_path=None,
        horizon_hours=None,
    )

    registry["baseline_last_value"] = ModelSpec(
        key="baseline_last_value",
        label="Baseline: Last value",
        family="Baseline",
        canonical_family="Baseline",
        canonical_model="Naive (last value)",
        artifact_model_path=None,
        manifest_path=None,
        horizon_hours=None,
    )

    registry["prophet"] = ModelSpec(
        key="prophet",
        label="Prophet (saved model, if installed)",
        family="Prophet",
        canonical_family="Prophet",
        canonical_model="Prophet",
        artifact_model_path=(root / "outputs" / "prophet_model.joblib")
        if (root / "outputs" / "prophet_model.joblib").exists()
        else None,
        manifest_path=(root / "outputs" / "manifests" / "prophet_prophet_h1.json")
        if (root / "outputs" / "manifests" / "prophet_prophet_h1.json").exists()
        else None,
        horizon_hours=1,
    )

    registry["arima"] = ModelSpec(
        key="arima",
        label="SARIMAX (saved model)",
        family="ARIMA",
        canonical_family="ARIMA",
        canonical_model="SARIMAX",
        artifact_model_path=(root / "outputs" / "arima_model.pkl") if (root / "outputs" / "arima_model.pkl").exists() else None,
        manifest_path=(root / "outputs" / "manifests" / "arima_sarimax_h1.json")
        if (root / "outputs" / "manifests" / "arima_sarimax_h1.json").exists()
        else None,
        horizon_hours=1,
    )

    return registry


def _ensure_hourly_index(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.duplicated(keep="first")]
    return s.asfreq("h")


def _time_features(ts: pd.Timestamp) -> dict[str, float]:
    hour = int(ts.hour)
    dow = int(ts.dayofweek)
    month = int(ts.month)
    quarter = int(((month - 1) // 3) + 1)
    is_weekend = float(1 if dow >= 5 else 0)
    is_business_hour = float(1 if (dow <= 4 and 9 <= hour < 17) else 0)
    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
    month_sin = float(np.sin(2 * np.pi * month / 12.0))
    month_cos = float(np.cos(2 * np.pi * month / 12.0))
    return {
        "hour": float(hour),
        "day_of_week": float(dow),
        "month": float(month),
        "quarter": float(quarter),
        "is_weekend": is_weekend,
        "is_business_hour": is_business_hour,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }


def _lag_value(s: pd.Series, ts: pd.Timestamp, hours: int) -> float:
    key = ts - pd.Timedelta(hours=int(hours))
    v = s.get(key, np.nan)
    return float(v) if pd.notna(v) else np.nan


def _rolling_stats(s: pd.Series, ts: pd.Timestamp, window_hours: int) -> tuple[float, float]:
    end = ts - pd.Timedelta(hours=1)
    start = ts - pd.Timedelta(hours=int(window_hours))
    window = s.loc[start:end].dropna()
    if len(window) < int(window_hours):
        return np.nan, np.nan
    arr = window.to_numpy(dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def build_ml_feature_row(ts: pd.Timestamp, y: pd.Series) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(_time_features(ts))

    for h in (1, 2, 3, 6, 12, 24):
        feats[f"lag_{h}h"] = _lag_value(y, ts, h)

    feats["same_hour_yesterday"] = _lag_value(y, ts, 24)
    feats["same_hour_last_week"] = _lag_value(y, ts, 168)

    for h in (6, 12, 24):
        mean_v, std_v = _rolling_stats(y, ts, h)
        feats[f"roll_mean_{h}h"] = mean_v
        feats[f"roll_std_{h}h"] = std_v

    return feats


def _infer_feature_cols_from_manifest(root: Path, spec: ModelSpec) -> list[str] | None:
    if spec.manifest_path is None or not spec.manifest_path.exists():
        return None
    try:
        manifest = load_json(spec.manifest_path)
    except Exception:
        return None
    cols = manifest.get("feature_cols")
    if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
        return [str(c) for c in cols]
    return None


def _resolve_ml_artifact_for_request(
    root: Path, spec: ModelSpec, requested_horizon: int, forecast_mode: str
) -> tuple[Path, Path | None, int]:
    wants_direct = forecast_mode == "Direct (single point at horizon)"
    if spec.horizon_to_model_path is not None:
        by_h = spec.horizon_to_model_path
        if wants_direct:
            if requested_horizon not in by_h:
                available = ", ".join(str(h) for h in sorted(by_h.keys()))
                raise RuntimeError(f"Model is only available for horizons: {available}h.")
            model_path = by_h[int(requested_horizon)]
            manifest = None
            if spec.horizon_to_manifest_path is not None:
                manifest = spec.horizon_to_manifest_path.get(int(requested_horizon))
            return model_path, manifest, int(requested_horizon)

        if 1 not in by_h:
            available = ", ".join(str(h) for h in sorted(by_h.keys()))
            raise RuntimeError(
                f"Recursive forecasts require a 1h model, but this selection only has: {available}h."
            )
        model_path = by_h[1]
        manifest = None
        if spec.horizon_to_manifest_path is not None:
            manifest = spec.horizon_to_manifest_path.get(1)
        return model_path, manifest, 1

    if spec.artifact_model_path is None:
        raise RuntimeError("Model artifact not available for this selection.")

    native_h = int(spec.horizon_hours) if spec.horizon_hours is not None else None
    if wants_direct:
        if native_h is not None and int(native_h) != int(requested_horizon):
            raise RuntimeError(
                f"Selected model was trained for h={native_h}h; choose h={native_h}h or select an auto-matching model."
            )
    else:
        if native_h is not None and int(native_h) != 1:
            raise RuntimeError(
                f"Recursive forecasts require a 1h model, but this selection was trained for h={native_h}h."
            )
    return spec.artifact_model_path, spec.manifest_path, int(native_h or requested_horizon)


def _infer_feature_cols_from_model(model: object) -> list[str] | None:
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        return None
    try:
        out = [str(c) for c in list(cols)]
        return out if out else None
    except Exception:
        return None


def forecast_randomforest_recursive(
    model: object,
    y_history: pd.Series,
    horizon_hours: int,
    feature_cols: list[str] | None,
) -> pd.DataFrame:
    y = _ensure_hourly_index(y_history.dropna())
    asof = pd.Timestamp(y.index.max())
    y_ext = y.copy()

    preds: list[tuple[pd.Timestamp, float]] = []
    for step in range(int(horizon_hours)):
        base_ts = asof + pd.Timedelta(hours=int(step))
        feats = build_ml_feature_row(base_ts, y_ext)
        X = pd.DataFrame([feats])
        if feature_cols is not None:
            X = X.reindex(columns=feature_cols)
        if np.any(~np.isfinite(X.to_numpy(dtype=float))):
            raise RuntimeError("Not enough history to compute ML features for this forecast horizon.")
        X_in = X if getattr(model, "feature_names_in_", None) is not None else X.to_numpy(dtype=float)
        y_hat = float(np.asarray(model.predict(X_in), dtype=float).ravel()[0])  # type: ignore[attr-defined]
        pred_ts = base_ts + pd.Timedelta(hours=1)
        preds.append((pred_ts, y_hat))
        y_ext.loc[pred_ts] = y_hat

    df = pd.DataFrame(preds, columns=["datetime", "yhat"])
    df["yhat_lower"] = np.nan
    df["yhat_upper"] = np.nan
    return df


def forecast_randomforest_direct(
    model: object,
    y_history: pd.Series,
    horizon_hours: int,
    feature_cols: list[str] | None,
) -> pd.DataFrame:
    y = _ensure_hourly_index(y_history.dropna())
    asof = pd.Timestamp(y.index.max())
    feats = build_ml_feature_row(asof, y)
    X = pd.DataFrame([feats])
    if feature_cols is not None:
        X = X.reindex(columns=feature_cols)
    if np.any(~np.isfinite(X.to_numpy(dtype=float))):
        raise RuntimeError("Not enough history to compute ML features for this horizon.")
    X_in = X if getattr(model, "feature_names_in_", None) is not None else X.to_numpy(dtype=float)
    y_hat = float(np.asarray(model.predict(X_in), dtype=float).ravel()[0])  # type: ignore[attr-defined]
    ts = asof + pd.Timedelta(hours=int(horizon_hours))
    df = pd.DataFrame({"datetime": [ts], "yhat": [y_hat], "yhat_lower": [np.nan], "yhat_upper": [np.nan]})
    return df


def forecast_sarimax(model: object, horizon_hours: int) -> pd.DataFrame:
    forecast_res = model.get_forecast(steps=int(horizon_hours))
    mean = forecast_res.predicted_mean
    ci = forecast_res.conf_int(alpha=0.05)
    idx = mean.index
    if not isinstance(idx, pd.DatetimeIndex):
        start = pd.Timestamp.utcnow().floor("h")
        idx = pd.date_range(start=start, periods=int(horizon_hours), freq="h")
    df = pd.DataFrame({"datetime": pd.to_datetime(idx), "yhat": mean.to_numpy(dtype=float)})
    if ci is not None and ci.shape[1] >= 2:
        df["yhat_lower"] = pd.to_numeric(ci.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        df["yhat_upper"] = pd.to_numeric(ci.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    else:
        df["yhat_lower"] = np.nan
        df["yhat_upper"] = np.nan
    return df


def _prophet_regressors(ds: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(ds, errors="coerce")
    hour = ts.dt.hour.astype(int)
    dow = ts.dt.dayofweek.astype(int)
    month = ts.dt.month.astype(int)
    quarter = (((month - 1) // 3) + 1).astype(int)
    is_weekend = (dow >= 5).astype(int)
    is_business_hour = ((dow <= 4) & (hour >= 9) & (hour < 17)).astype(int)
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    return pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": dow,
            "month": month,
            "quarter": quarter,
            "is_weekend": is_weekend,
            "is_business_hour": is_business_hour,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
    )


def forecast_prophet(model: object, horizon_hours: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=int(horizon_hours), freq="h", include_history=False)  # type: ignore[attr-defined]
    regs = _prophet_regressors(future["ds"])
    for c in regs.columns:
        future[c] = regs[c]
    fcst = model.predict(future)  # type: ignore[attr-defined]
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(fcst["ds"], errors="coerce"),
            "yhat": pd.to_numeric(fcst["yhat"], errors="coerce"),
            "yhat_lower": pd.to_numeric(fcst.get("yhat_lower"), errors="coerce"),
            "yhat_upper": pd.to_numeric(fcst.get("yhat_upper"), errors="coerce"),
        }
    )
    return df.dropna(subset=["datetime", "yhat"]).reset_index(drop=True)


def forecast_baseline(df_all: pd.DataFrame, horizon_hours: int, variant: str) -> pd.DataFrame:
    s = make_hourly_series(df_all).dropna()
    asof = pd.Timestamp(s.index.max())
    future_idx = pd.date_range(start=asof + pd.Timedelta(hours=1), periods=int(horizon_hours), freq="h")
    if variant == "Naive (last value)":
        yhat = np.full(len(future_idx), float(s.iloc[-1]), dtype=float)
    elif variant == "Seasonal naive (daily)":
        yhat = s.shift(24).reindex(future_idx).to_numpy(dtype=float)
    elif variant == "Seasonal naive (weekly)":
        yhat = s.shift(168).reindex(future_idx).to_numpy(dtype=float)
    elif variant == "Moving average (24)":
        yhat = s.rolling(window=24, min_periods=1).mean().iloc[-1]
        yhat = np.full(len(future_idx), float(yhat), dtype=float)
    elif variant == "Moving average (48)":
        yhat = s.rolling(window=48, min_periods=1).mean().iloc[-1]
        yhat = np.full(len(future_idx), float(yhat), dtype=float)
    else:
        yhat = np.full(len(future_idx), np.nan, dtype=float)
    df = pd.DataFrame({"datetime": future_idx, "yhat": yhat, "yhat_lower": np.nan, "yhat_upper": np.nan})
    return df.dropna(subset=["datetime", "yhat"])


def export_plot_html(fig: go.Figure) -> bytes:
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    return html.encode("utf-8")


def export_plot_png(fig: go.Figure) -> bytes | None:
    try:
        return pio.to_image(fig, format="png", width=1400, height=700, scale=2)
    except Exception:
        return None


def render_downloads(forecast_df: pd.DataFrame, fig: go.Figure, filename_prefix: str) -> None:
    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forecast CSV",
        data=csv_bytes,
        file_name=f"{filename_prefix}_forecast.csv",
        mime="text/csv",
        **_width_kwargs(st.download_button, True),
    )
    st.download_button(
        "Download plot HTML",
        data=export_plot_html(fig),
        file_name=f"{filename_prefix}_plot.html",
        mime="text/html",
        **_width_kwargs(st.download_button, True),
    )
    png = export_plot_png(fig)
    if png is not None:
        st.download_button(
            "Download plot PNG",
            data=png,
            file_name=f"{filename_prefix}_plot.png",
            mime="image/png",
            **_width_kwargs(st.download_button, True),
        )
    else:
        st.caption("PNG export requires plotly+kaleido. Install kaleido to enable PNG downloads.")


def render_backtest_downloads(df_pred: pd.DataFrame, fig: go.Figure, filename_prefix: str) -> None:
    csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download backtest CSV",
        data=csv_bytes,
        file_name=f"{filename_prefix}_backtest.csv",
        mime="text/csv",
        **_width_kwargs(st.download_button, True),
    )
    st.download_button(
        "Download backtest plot HTML",
        data=export_plot_html(fig),
        file_name=f"{filename_prefix}_backtest_plot.html",
        mime="text/html",
        **_width_kwargs(st.download_button, True),
    )
    png = export_plot_png(fig)
    if png is not None:
        st.download_button(
            "Download backtest plot PNG",
            data=png,
            file_name=f"{filename_prefix}_backtest_plot.png",
            mime="image/png",
            **_width_kwargs(st.download_button, True),
        )
    else:
        st.caption("PNG export requires plotly+kaleido. Install kaleido to enable PNG downloads.")


def inject_css() -> None:
    st.markdown(
        """
        <style>
          :root {
            --bg: #0b1220;
            --panel: #111a2e;
            --panel-2: #0f172a;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-2: #a78bfa;
            --danger: #fb7185;
            --border: rgba(148, 163, 184, 0.18);
          }

          .stApp {
            background: radial-gradient(1200px 700px at 20% 0%, rgba(56,189,248,0.10), transparent 60%),
                        radial-gradient(900px 600px at 90% 10%, rgba(167,139,250,0.10), transparent 55%),
                        var(--bg);
            color: var(--text);
          }

          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(17,26,46,0.98), rgba(15,23,42,0.98));
            border-right: 1px solid var(--border);
          }

          .stButton > button {
            border-radius: 10px;
            border: 1px solid var(--border);
          }

          .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid var(--border);
          }

          div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(17,26,46,0.92), rgba(15,23,42,0.92));
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 10px 12px;
          }

          div[data-testid="stMetric"] label {
            color: var(--muted) !important;
          }

          @media (max-width: 900px) {
            .block-container {
              padding-left: 0.8rem;
              padding-right: 0.8rem;
            }
          }

          @media (max-width: 600px) {
            div[data-testid="stMetric"] {
              padding: 8px 10px;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )



def render_main(df_all: pd.DataFrame, inputs: SidebarInputs) -> None:
    st.title("Energy Consumption Forecasting Dashboard")
    st.write(
        "Explore historical energy consumption and generate short-term forecasts using saved models and fast baselines."
    )

    if pd.Timestamp(inputs.start_date) > pd.Timestamp(inputs.end_date):
        st.error("Invalid date range: start date must be on or before end date.")
        st.stop()

    df_hist = filter_date_range(df_all, inputs.start_date, inputs.end_date)
    metrics = compute_key_metrics(df_hist)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest consumption (kW)", value=f"{metrics['latest_kw']:.3f}" if np.isfinite(metrics["latest_kw"]) else "—")
    c2.metric(
        "Avg daily usage (kWh)",
        value=f"{metrics['avg_daily_kwh']:.1f}" if np.isfinite(metrics["avg_daily_kwh"]) else "—",
    )
    c3.metric("Peak (kW)", value=f"{metrics['peak_kw']:.3f}" if np.isfinite(metrics["peak_kw"]) else "—")
    c4.metric(
        "Last 24h usage (kWh)",
        value=f"{metrics['last_24h_kwh']:.1f}" if np.isfinite(metrics["last_24h_kwh"]) else "—",
    )

    st.subheader("Historical Consumption")
    if df_hist.empty:
        st.info("No data available for the selected date range.")
    else:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=df_hist["datetime"],
                y=df_hist["Global_active_power"],
                mode="lines",
                name="Actual",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="%{x}<br>%{y:.3f} kW<extra></extra>",
            )
        )
        fig_hist.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_hist.update_xaxes(
            title_text="Datetime",
            rangeslider=dict(visible=True),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
        )
        fig_hist.update_yaxes(title_text="Consumption (kW)", fixedrange=False)
        st.plotly_chart(fig_hist, use_container_width=True, config={"responsive": True})

    st.subheader("Seasonal Patterns")
    if df_hist.empty:
        st.info("Seasonal views require historical data in the selected range.")
    else:
        heatmap_tab, box_tab = st.tabs(["Heatmap (Hour x Day)", "Box Plots"])

        with heatmap_tab:
            pivot = build_seasonal_heatmap(df_hist)
            if pivot.empty:
                st.info("Not enough data to build a heatmap for this range.")
            else:
                fig_heatmap = px.imshow(
                    pivot,
                    labels=dict(x="Day of week", y="Hour of day", color="Avg kW"),
                    aspect="auto",
                    color_continuous_scale="Viridis",
                )
                fig_heatmap.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_heatmap, use_container_width=True, config={"responsive": True})

        with box_tab:
            grouping = st.selectbox(
                "Group by",
                options=["Hour of day", "Day of week", "Month", "Weekend vs weekday"],
                index=1,
            )
            df_box = df_hist[["datetime", "Global_active_power"]].copy()
            df_box["hour"] = df_box["datetime"].dt.hour
            df_box["dow"] = df_box["datetime"].dt.dayofweek
            df_box["month"] = df_box["datetime"].dt.month
            df_box["is_weekend"] = df_box["dow"].isin([5, 6])

            if grouping == "Hour of day":
                df_box["group"] = df_box["hour"].astype(str)
                order = [str(i) for i in range(24)]
                title = "Distribution by hour"
            elif grouping == "Day of week":
                df_box["group"] = df_box["dow"].map(
                    {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
                )
                order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                title = "Distribution by day of week"
            elif grouping == "Month":
                df_box["group"] = df_box["month"].astype(str)
                order = [str(i) for i in range(1, 13)]
                title = "Distribution by month"
            else:
                df_box["group"] = df_box["is_weekend"].map({False: "Weekday", True: "Weekend"})
                order = ["Weekday", "Weekend"]
                title = "Distribution by weekend vs weekday"

            fig_box = px.box(
                df_box,
                x="group",
                y="Global_active_power",
                category_orders={"group": order},
                points="outliers",
                labels={"group": "", "Global_active_power": "Consumption (kW)"},
                title=title,
            )
            fig_box.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig_box, use_container_width=True, config={"responsive": True})

    st.subheader("Forecast")
    if inputs.compare_mode:
        st.caption(f"Comparison mode | Models: {len(inputs.selected_models)} | Horizon: {inputs.horizon_hours}h")
    else:
        root = get_project_root()
        reg = build_model_registry(root)
        sel = inputs.selected_models[0] if inputs.selected_models else ""
        label = reg.get(sel).label if sel in reg else (sel or "—")
        st.caption(f"Selected model: {label} | Horizon: {inputs.horizon_hours}h")

    should_run = bool(inputs.generate_clicked or inputs.auto_generate)
    if inputs.refresh_seconds > 0 and (inputs.realtime_mode or inputs.auto_generate):
        _maybe_autorefresh(True, inputs.refresh_seconds)

    if should_run:
        root = get_project_root()
        comparison_path = root / "outputs" / "comparison" / "model_comparison_all.csv"
        model_comparison = pd.DataFrame()
        if comparison_path.exists():
            try:
                model_comparison = load_model_comparison(comparison_path)
            except Exception:
                model_comparison = pd.DataFrame()

        baseline_ref_row = resolve_baseline_reference_row(model_comparison, inputs.horizon_hours, inputs.baseline_reference)

        start = pd.Timestamp(inputs.start_date).normalize()
        end = pd.Timestamp(inputs.end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        df_pred_model: pd.DataFrame | None = None
        model_label = inputs.selected_models[0] if inputs.selected_models else "model"
        artifact_row = None
        if not inputs.compare_mode and inputs.selected_models:
            spec_key = inputs.selected_models[0]
            reg = build_model_registry(root)
            spec = reg.get(spec_key)
            if spec is not None and model_comparison is not None and not model_comparison.empty:
                family = str(spec.canonical_family or spec.family)
                model_name = str(spec.canonical_model or spec.label).split(":")[-1].split("(")[0].strip()
                candidates = model_comparison[
                    (model_comparison["Family"] == family)
                    & (model_comparison["Model"] == model_name)
                    & (model_comparison["Horizon_h"] == int(inputs.horizon_hours))
                ]
                if not candidates.empty:
                    artifact_row = candidates.iloc[0]
                    predictions_path = root / str(artifact_row["Predictions"])
                    if predictions_path.exists():
                        df_pred = load_predictions(predictions_path).copy()
                        if {"datetime", "y_true", "y_pred"}.issubset(set(df_pred.columns)):
                            df_pred["y_true"] = pd.to_numeric(df_pred["y_true"], errors="coerce")
                            df_pred["y_pred"] = pd.to_numeric(df_pred["y_pred"], errors="coerce")
                            df_pred = df_pred.dropna(subset=["datetime", "y_true", "y_pred"]).sort_values("datetime")
                            df_pred_model = df_pred
                            model_label = model_name

        if df_pred_model is not None and not df_pred_model.empty:
            model_metrics = compute_error_metrics(df_pred_model["y_true"].to_numpy(), df_pred_model["y_pred"].to_numpy())

            baseline_label = inputs.baseline_reference
            baseline_metrics: dict[str, float] | None = None
            if baseline_ref_row is not None and baseline_label == "Auto (best available)":
                baseline_label = str(baseline_ref_row["Model"])
                baseline_metrics = {
                    "mae": float(baseline_ref_row["MAE"]),
                    "rmse": float(baseline_ref_row["RMSE"]),
                    "mape_pct": float(baseline_ref_row["MAPE"]),
                }

            if baseline_metrics is None:
                if baseline_label == "Auto (best available)":
                    if inputs.horizon_hours >= 168:
                        baseline_label = "Seasonal naive (weekly)"
                    elif inputs.horizon_hours >= 24:
                        baseline_label = "Seasonal naive (daily)"
                    else:
                        baseline_label = "Naive (last value)"

                if baseline_label in BASELINE_VARIANTS:
                    s_hourly = make_hourly_series(df_all)
                    baseline_pred = baseline_predictions_for_timestamps(
                        s_hourly, df_pred_model["datetime"], baseline_label, inputs.horizon_hours
                    )
                    baseline_metrics = compute_error_metrics(
                        df_pred_model["y_true"].to_numpy(), baseline_pred.to_numpy()
                    )
                else:
                    baseline_metrics = {"mae": np.nan, "rmse": np.nan, "mape_pct": np.nan}

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "MAE",
                value=f"{model_metrics['mae']:.4f}" if np.isfinite(model_metrics["mae"]) else "—",
                delta=f"{format_pct(compute_improvement_pct(baseline_metrics['mae'], model_metrics['mae']))} vs {baseline_label}",
            )
            m2.metric(
                "RMSE",
                value=f"{model_metrics['rmse']:.4f}" if np.isfinite(model_metrics["rmse"]) else "—",
                delta=f"{format_pct(compute_improvement_pct(baseline_metrics['rmse'], model_metrics['rmse']))} vs {baseline_label}",
            )
            m3.metric(
                "MAPE (%)",
                value=f"{model_metrics['mape_pct']:.2f}" if np.isfinite(model_metrics["mape_pct"]) else "—",
                delta=f"{format_pct(compute_improvement_pct(baseline_metrics['mape_pct'], model_metrics['mape_pct']))} vs {baseline_label}",
            )

        forecast_tab, backtest_tab = st.tabs(["Future Forecast (Live)", "Actual vs Predicted (Backtest)"])

        with forecast_tab:
            with st.spinner("Generating forecast..."):
                forecasts: dict[str, pd.DataFrame] = {}
                errors: dict[str, str] = {}
                registry = build_model_registry(root)

                history_series = make_hourly_series(df_all).dropna()
                if inputs.realtime_mode:
                    if "stream_index" not in st.session_state:
                        st.session_state["stream_index"] = int(len(history_series) * 0.90)
                    if inputs.refresh_seconds > 0:
                        st.session_state["stream_index"] = min(
                            len(history_series) - 1,
                            int(st.session_state["stream_index"]) + int(inputs.stream_speed_hours),
                        )
                    base_series = history_series.iloc[: int(st.session_state["stream_index"]) + 1]
                else:
                    base_series = history_series

                if not inputs.selected_models:
                    st.warning("Select at least one model in the sidebar.")
                    st.stop()

                if inputs.realtime_mode:
                    c1, c2 = st.columns([3, 1])
                    c1.caption(f"Streaming as-of: {pd.Timestamp(base_series.index.max())}")
                    if c2.button("Reset stream", **_width_kwargs(c2.button, True)):
                        st.session_state["stream_index"] = int(len(history_series) * 0.90)
                        st.rerun()

                for key in inputs.selected_models:
                    spec = registry.get(key)
                    if spec is None:
                        continue
                    try:
                        if key in {"baseline_last_value", "baseline_moving_average", "baseline_seasonal_naive"}:
                            if key == "baseline_seasonal_naive":
                                variant = inputs.seasonal_naive_variant
                            elif key == "baseline_moving_average":
                                variant = inputs.moving_average_variant
                            else:
                                variant = "Naive (last value)"
                            forecasts[key] = forecast_baseline(df_all, inputs.horizon_hours, variant)
                            continue

                        if key == "prophet":
                            if spec.artifact_model_path is None:
                                raise RuntimeError("Saved Prophet model not found.")
                            model = load_prophet_model(spec.artifact_model_path)
                            forecasts[key] = forecast_prophet(model, inputs.horizon_hours)
                            continue

                        if key == "arima":
                            if spec.artifact_model_path is None:
                                raise RuntimeError("Saved SARIMAX model not found.")
                            model = load_sarimax_results(spec.artifact_model_path)
                            forecasts[key] = forecast_sarimax(model, inputs.horizon_hours)
                            continue

                        if spec.artifact_model_path is None:
                            if spec.horizon_to_model_path is None:
                                raise RuntimeError("Model artifact not available for this selection.")

                        model_path, manifest_path, native_h = _resolve_ml_artifact_for_request(
                            root, spec, int(inputs.horizon_hours), str(inputs.forecast_mode)
                        )
                        spec_for_infer = ModelSpec(
                            key=spec.key,
                            label=spec.label,
                            family=spec.family,
                            canonical_family=spec.canonical_family,
                            canonical_model=spec.canonical_model,
                            artifact_model_path=model_path,
                            manifest_path=manifest_path,
                            horizon_hours=native_h,
                        )
                        model = load_joblib_model(model_path)
                        feature_cols = _infer_feature_cols_from_model(model) or _infer_feature_cols_from_manifest(root, spec_for_infer)

                        if inputs.forecast_mode == "Direct (single point at horizon)":
                            forecasts[key] = forecast_randomforest_direct(model, base_series, int(inputs.horizon_hours), feature_cols)
                        else:
                            forecasts[key] = forecast_randomforest_recursive(model, base_series, int(inputs.horizon_hours), feature_cols)
                    except Exception as e:
                        errors[key] = str(e)

            if not model_comparison.empty:
                with st.expander("Model performance (from backtests)", expanded=False):
                    keys = set(inputs.selected_models)
                    rows = []
                    registry = build_model_registry(root)
                    for k in keys:
                        spec = registry.get(k)
                        if spec is None:
                            continue
                        family = str(spec.canonical_family or spec.family)
                        if k == "baseline_seasonal_naive":
                            model_name = str(inputs.seasonal_naive_variant)
                        elif k == "baseline_moving_average":
                            model_name = str(inputs.moving_average_variant)
                        elif k == "baseline_last_value":
                            model_name = "Naive (last value)"
                        else:
                            model_name = str(spec.canonical_model or spec.label).split(":")[-1].split("(")[0].strip()
                        matches = model_comparison[
                            (model_comparison["Family"] == family) & (model_comparison["Model"] == model_name)
                        ].copy()
                        if matches.empty:
                            continue
                        matches = matches.sort_values("Horizon_h")
                        for _, r in matches.iterrows():
                            rows.append(
                                {
                                    "Selection": registry.get(k).label if registry.get(k) else k,
                                    "Horizon_h": int(r["Horizon_h"]),
                                    "MAE": float(r["MAE"]),
                                    "RMSE": float(r["RMSE"]),
                                    "MAPE": float(r["MAPE"]),
                                }
                            )
                    if rows:
                        perf_df = pd.DataFrame(rows).sort_values(["Selection", "Horizon_h"])
                        st.dataframe(perf_df, hide_index=True, **_width_kwargs(st.dataframe, True))
                        st.download_button(
                            "Download performance table CSV",
                            data=perf_df.to_csv(index=False).encode("utf-8"),
                            file_name="model_performance_table.csv",
                            mime="text/csv",
                            **_width_kwargs(st.download_button, True),
                        )

            if errors:
                for k, msg in errors.items():
                    label = registry.get(k).label if registry.get(k) else k
                    st.error(f"{label}: {msg}")

            if not forecasts:
                st.warning("No forecasts could be generated with the selected models.")
            else:
                merged = None
                for k, fc in forecasts.items():
                    label = registry.get(k).label if registry.get(k) else k
                    frame = fc[["datetime", "yhat"]].rename(columns={"yhat": label})
                    merged = frame if merged is None else merged.merge(frame, on="datetime", how="outer")
                merged = merged.sort_values("datetime") if merged is not None else pd.DataFrame()

                views = (
                    st.tabs(["Overlay", "Side-by-side"]) if len(forecasts) > 1 else (st.container(), None)
                )
                overlay_view = views[0]
                side_view = views[1]

                with overlay_view:
                    fig_future = go.Figure()
                    hist_view = df_all.sort_values("datetime").tail(7 * 24)
                    fig_future.add_trace(
                        go.Scatter(
                            x=hist_view["datetime"],
                            y=hist_view["Global_active_power"],
                            mode="lines",
                            name="Actual (last 7d)",
                            line=dict(color="#1f77b4", width=2),
                            hovertemplate="%{x}<br>%{y:.3f} kW<extra></extra>",
                        )
                    )

                    palette = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
                    for i, (k, fc) in enumerate(forecasts.items()):
                        label = registry.get(k).label if registry.get(k) else k
                        fig_future.add_trace(
                            go.Scatter(
                                x=fc["datetime"],
                                y=fc["yhat"],
                                mode="lines",
                                name=label,
                                line=dict(color=palette[i % len(palette)], width=2, dash="dash" if i == 0 else "solid"),
                                hovertemplate="%{x}<br>%{y:.3f} kW<extra></extra>",
                            )
                        )
                        if {"yhat_lower", "yhat_upper"}.issubset(set(fc.columns)) and fc["yhat_lower"].notna().any():
                            fig_future.add_trace(
                                go.Scatter(
                                    x=fc["datetime"],
                                    y=fc["yhat_upper"],
                                    mode="lines",
                                    line=dict(width=0),
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )
                            fig_future.add_trace(
                                go.Scatter(
                                    x=fc["datetime"],
                                    y=fc["yhat_lower"],
                                    mode="lines",
                                    line=dict(width=0),
                                    fill="tonexty",
                                    fillcolor="rgba(255, 127, 14, 0.12)",
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )

                    fig_future.update_layout(
                        height=420,
                        margin=dict(l=10, r=10, t=30, b=10),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    fig_future.update_xaxes(title_text="Datetime", showspikes=True, spikemode="across", spikesnap="cursor")
                    fig_future.update_yaxes(title_text="Consumption (kW)", fixedrange=False)
                    st.plotly_chart(fig_future, use_container_width=True, config={"responsive": True})

                    st.dataframe(merged, hide_index=True, **_width_kwargs(st.dataframe, True))

                    if len(forecasts) == 1:
                        key = next(iter(forecasts))
                        render_downloads(forecasts[key], fig_future, filename_prefix=str(key))
                    else:
                        combined_bytes = merged.to_csv(index=False).encode("utf-8") if merged is not None else b""
                        st.download_button(
                            "Download combined forecast CSV",
                            data=combined_bytes,
                            file_name="combined_forecast.csv",
                            mime="text/csv",
                            **_width_kwargs(st.download_button, True),
                        )
                        st.download_button(
                            "Download combined plot HTML",
                            data=export_plot_html(fig_future),
                            file_name="combined_forecast_plot.html",
                            mime="text/html",
                            **_width_kwargs(st.download_button, True),
                        )
                        png = export_plot_png(fig_future)
                        if png is not None:
                            st.download_button(
                                "Download combined plot PNG",
                                data=png,
                                file_name="combined_forecast_plot.png",
                                mime="image/png",
                                **_width_kwargs(st.download_button, True),
                            )
                        else:
                            st.caption("PNG export requires plotly+kaleido. Install kaleido to enable PNG downloads.")

                if side_view is not None:
                    with side_view:
                        items = list(forecasts.items())
                        cols = st.columns(2)
                        hist_view = df_all.sort_values("datetime").tail(7 * 24)
                        for i, (k, fc) in enumerate(items):
                            with cols[i % 2]:
                                label = registry.get(k).label if registry.get(k) else k
                                fig = go.Figure()
                                fig.add_trace(
                                    go.Scatter(
                                        x=hist_view["datetime"],
                                        y=hist_view["Global_active_power"],
                                        mode="lines",
                                        name="Actual (last 7d)",
                                        line=dict(color="#1f77b4", width=2),
                                        hovertemplate="%{x}<br>%{y:.3f} kW<extra></extra>",
                                    )
                                )
                                fig.add_trace(
                                    go.Scatter(
                                        x=fc["datetime"],
                                        y=fc["yhat"],
                                        mode="lines",
                                        name=label,
                                        line=dict(color="#ff7f0e", width=2),
                                        hovertemplate="%{x}<br>%{y:.3f} kW<extra></extra>",
                                    )
                                )
                                fig.update_layout(
                                    height=330,
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    hovermode="x unified",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                )
                                fig.update_xaxes(title_text="Datetime")
                                fig.update_yaxes(title_text="Consumption (kW)", fixedrange=False)
                                st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
                                st.dataframe(fc[["datetime", "yhat"]], hide_index=True, **_width_kwargs(st.dataframe, True))
                                render_downloads(fc, fig, filename_prefix=str(k))

        with backtest_tab:
            if df_pred_model is None or df_pred_model.empty:
                st.info("No predictions available for this model/horizon. Run the training scripts to generate outputs.")
            else:
                df_pred_view = df_pred_model[(df_pred_model["datetime"] >= start) & (df_pred_model["datetime"] <= end)]
                if df_pred_view.empty:
                    df_pred_view = df_pred_model

                fig_backtest = go.Figure()
                fig_backtest.add_trace(
                    go.Scatter(
                        x=df_pred_view["datetime"],
                        y=df_pred_view["y_true"],
                        mode="lines",
                        name="Actual",
                        line=dict(color="#1f77b4", width=2),
                        hovertemplate="%{x}<br>%{y:.3f} kW<extra></extra>",
                    )
                )
                fig_backtest.add_trace(
                    go.Scatter(
                        x=df_pred_view["datetime"],
                        y=df_pred_view["y_pred"],
                        mode="lines",
                        name="Predicted",
                        line=dict(color="#ff7f0e", width=2),
                        hovertemplate="%{x}<br>%{y:.3f} kW<extra></extra>",
                    )
                )

                df_ci = load_ci_forecast_if_available(root, inputs)
                if df_ci is not None and {"ds", "yhat", "yhat_lower", "yhat_upper"}.issubset(set(df_ci.columns)):
                    df_ci_view = df_ci.dropna(subset=["ds"]).sort_values("ds")
                    df_ci_view = df_ci_view[
                        (df_ci_view["ds"] >= df_pred_view["datetime"].min())
                        & (df_ci_view["ds"] <= df_pred_view["datetime"].max())
                    ]
                    if not df_ci_view.empty:
                        fig_backtest.add_trace(
                            go.Scatter(
                                x=df_ci_view["ds"],
                                y=df_ci_view["yhat_upper"],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                        fig_backtest.add_trace(
                            go.Scatter(
                                x=df_ci_view["ds"],
                                y=df_ci_view["yhat_lower"],
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor="rgba(255, 127, 14, 0.18)",
                                name="Confidence interval",
                                hoverinfo="skip",
                            )
                        )

                if artifact_row is not None:
                    annotation_text = (
                        f"{model_label} | h={int(artifact_row['Horizon_h'])}"
                        f"<br>MAE={artifact_row['MAE']:.4f} RMSE={artifact_row['RMSE']:.4f} MAPE={artifact_row['MAPE']:.2f}%"
                    )
                else:
                    metrics = compute_error_metrics(df_pred_model["y_true"].to_numpy(), df_pred_model["y_pred"].to_numpy())
                    annotation_text = (
                        f"{model_label} | h={inputs.horizon_hours}"
                        f"<br>MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} MAPE={metrics['mape_pct']:.2f}%"
                    )

                fig_backtest.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=30, b=10),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    annotations=[
                        dict(
                            xref="paper",
                            yref="paper",
                            x=0.01,
                            y=0.99,
                            xanchor="left",
                            yanchor="top",
                            text=annotation_text,
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.7)",
                            bordercolor="rgba(0,0,0,0.15)",
                            borderwidth=1,
                        )
                    ],
                )
                fig_backtest.update_xaxes(
                    title_text="Datetime",
                    rangeslider=dict(visible=True),
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                )
                fig_backtest.update_yaxes(title_text="Consumption (kW)", fixedrange=False)
                st.plotly_chart(fig_backtest, use_container_width=True, config={"responsive": True})
                st.dataframe(df_pred_view, hide_index=True, **_width_kwargs(st.dataframe, True))
                render_backtest_downloads(df_pred_view, fig_backtest, filename_prefix=f"{model_label}_h{inputs.horizon_hours}")
    else:
        st.info("Use the sidebar to choose a model and click Generate Forecast.")


def main() -> None:
    st.set_page_config(page_title="Energy Forecasting", page_icon="⚡", layout="wide")
    inject_css()
    pio.templates.default = "plotly_dark"

    root = get_project_root()
    default_csv = root / "data" / "processed" / "global_active_power_hourly.csv"
    data_csv = _path_from_env_or_default("ENERGY_APP_DATA_CSV", default_csv)

    try:
        df_all = load_hourly_series(data_csv)
    except Exception as e:
        st.error(f"Failed to load data from {data_csv}.")
        if _debug_enabled():
            st.exception(e)
        else:
            st.caption("Set ENERGY_APP_DEBUG=1 to show the full error traceback.")
        st.stop()

    inputs = render_sidebar(df_all)
    render_main(df_all, inputs)


if __name__ == "__main__":
    main()

