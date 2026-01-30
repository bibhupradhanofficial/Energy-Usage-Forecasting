from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    model_name: str
    y_pred: pd.Series
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any]


def _as_series(y: pd.Series | pd.DataFrame, name: str | None = None) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("DataFrame must have exactly one column to be treated as a series.")
        s = y.iloc[:, 0]
    else:
        s = y
    s = s.copy()
    if name is not None:
        s.name = name
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("Series must be indexed by a DatetimeIndex.")
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


def infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError("Need at least two timestamps to infer step.")
    diffs = pd.Series(index[1:] - index[:-1]).dropna()
    if diffs.empty:
        raise ValueError("Could not infer step from index.")
    return diffs.mode().iloc[0]


def seasonal_candidates_for_step(step: pd.Timedelta) -> dict[str, int]:
    one_hour = pd.Timedelta(hours=1)
    one_day = pd.Timedelta(days=1)
    if step == one_hour:
        return {"daily": 24, "weekly": 24 * 7}
    if step == one_day:
        return {"weekly": 7, "yearly": 365}
    return {}


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(np.mean(np.abs(y_true.to_numpy() - y_pred.to_numpy())))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    err = y_true.to_numpy() - y_pred.to_numpy()
    return float(np.sqrt(np.mean(err * err)))


def mape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-8) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    denom = np.abs(y_true.to_numpy())
    mask = denom > eps
    if not np.any(mask):
        return float("nan")
    out = np.mean(np.abs((y_true.to_numpy()[mask] - y_pred.to_numpy()[mask]) / denom[mask])) * 100.0
    return float(out)


def evaluate_forecast(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    metadata: Mapping[str, Any] | None = None,
) -> ForecastResult:
    y_true = _as_series(y_true, name="y_true")
    y_pred = _as_series(y_pred, name="y_pred")
    y_true, y_pred = y_true.align(y_pred, join="inner")
    metrics = {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred)}
    return ForecastResult(
        model_name=model_name,
        y_pred=y_pred,
        metrics=metrics,
        metadata=dict(metadata or {}),
    )


def naive_forecast(y_train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    y_train = _as_series(y_train, name="y_train")
    if len(y_train) == 0:
        raise ValueError("y_train is empty.")
    last = float(y_train.iloc[-1])
    out = pd.Series(np.full(shape=len(test_index), fill_value=last, dtype="float64"), index=test_index, name="naive")
    return out


def seasonal_naive_forecast(
    y_train: pd.Series,
    test_index: pd.DatetimeIndex,
    seasonal_periods: int,
) -> pd.Series:
    y_train = _as_series(y_train, name="y_train")
    step = infer_step(y_train.index)
    offset = seasonal_periods * step
    last_value = float(y_train.iloc[-1])
    values: list[float] = []
    for t in test_index:
        key = t - offset
        if key in y_train.index:
            values.append(float(y_train.loc[key]))
        else:
            values.append(last_value)
    return pd.Series(values, index=test_index, name=f"seasonal_naive_{seasonal_periods}")


def moving_average_forecast(
    y_train: pd.Series,
    test_index: pd.DatetimeIndex,
    window: int,
) -> pd.Series:
    y_train = _as_series(y_train, name="y_train")
    if window <= 0:
        raise ValueError("window must be > 0.")
    history = y_train.to_list()
    preds: list[float] = []
    for _ in range(len(test_index)):
        tail = history[-window:] if len(history) >= window else history
        pred = float(np.mean(tail))
        preds.append(pred)
        history.append(pred)
    return pd.Series(preds, index=test_index, name=f"moving_avg_{window}")


def arima_auto_forecast(
    y_train: pd.Series,
    test_index: pd.DatetimeIndex,
    seasonal: bool,
    m: int | None = None,
    max_p: int = 1,
    max_q: int = 1,
    max_d: int = 1,
    max_P: int = 1,
    max_Q: int = 1,
    max_D: int = 1,
    maxiter: int = 15,
) -> tuple[pd.Series, Mapping[str, Any]]:
    y_train = _as_series(y_train, name="y_train")
    import pmdarima as pm

    if seasonal and (m is None or m <= 1):
        raise ValueError("For seasonal ARIMA, m must be provided and > 1.")

    model = pm.auto_arima(
        y_train.to_numpy(),
        seasonal=seasonal,
        m=int(m) if seasonal else 1,
        start_p=0,
        start_q=0,
        max_p=max_p,
        max_q=max_q,
        max_d=max_d,
        start_P=0,
        start_Q=0,
        max_P=max_P,
        max_Q=max_Q,
        max_D=max_D,
        max_order=3,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        trace=False,
        method="lbfgs",
        maxiter=int(maxiter),
    )

    preds = model.predict(n_periods=len(test_index))
    meta = {
        "order": tuple(int(x) for x in model.order),
        "seasonal_order": tuple(int(x) for x in model.seasonal_order) if seasonal else None,
        "aic": float(model.aic()),
    }
    name = "SARIMA" if seasonal else "ARIMA"
    return pd.Series(preds.astype("float64"), index=test_index, name=name), meta


def holt_winters_forecast(
    y_train: pd.Series,
    test_index: pd.DatetimeIndex,
    seasonal_periods: int | None,
    trend: str | None = "add",
    seasonal: str | None = "add",
    damped_trend: bool = True,
    use_brute: bool = False,
) -> tuple[pd.Series, Mapping[str, Any]]:
    y_train = _as_series(y_train, name="y_train")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if seasonal_periods is None or seasonal is None:
        seasonal_periods = None
        seasonal = None

    model = ExponentialSmoothing(
        y_train,
        trend=trend,
        damped_trend=damped_trend if trend is not None else False,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True, use_brute=bool(use_brute))
    preds = fit.forecast(steps=len(test_index)).astype("float64")
    preds.index = test_index
    meta = {
        "trend": trend,
        "seasonal": seasonal,
        "seasonal_periods": seasonal_periods,
        "aic": float(getattr(fit, "aic", np.nan)),
    }
    return preds.rename("HoltWinters"), meta


def seasonal_strength(y: pd.Series, seasonal_periods: int) -> float:
    y = _as_series(y, name="y")
    if seasonal_periods <= 1 or len(y) <= seasonal_periods:
        return float("nan")
    try:
        return float(y.autocorr(lag=seasonal_periods))
    except Exception:
        return float("nan")
