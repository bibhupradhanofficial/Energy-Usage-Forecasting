from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class DieboldMarianoResult:
    statistic: float
    p_value: float
    n: int
    h: int


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _newey_west_variance(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 3:
        return float("nan")
    d0 = d - float(np.mean(d))
    gamma0 = float(np.mean(d0 * d0))
    if lag <= 0:
        return gamma0
    var = gamma0
    for k in range(1, int(lag) + 1):
        w = 1.0 - (k / (lag + 1.0))
        cov = float(np.mean(d0[k:] * d0[:-k]))
        var += 2.0 * w * cov
    return float(var)


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    h: int = 1,
    loss: Literal["se", "ae"] = "se",
) -> DieboldMarianoResult:
    y_true = np.asarray(y_true, dtype=float)
    y_pred_1 = np.asarray(y_pred_1, dtype=float)
    y_pred_2 = np.asarray(y_pred_2, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred_1) & np.isfinite(y_pred_2)
    y_true = y_true[mask]
    y_pred_1 = y_pred_1[mask]
    y_pred_2 = y_pred_2[mask]

    n = int(y_true.size)
    h = int(h)
    if n < 10:
        return DieboldMarianoResult(statistic=float("nan"), p_value=float("nan"), n=n, h=h)

    e1 = y_true - y_pred_1
    e2 = y_true - y_pred_2
    if loss == "ae":
        d = np.abs(e1) - np.abs(e2)
    else:
        d = (e1 * e1) - (e2 * e2)

    lag = max(0, h - 1)
    var_d = _newey_west_variance(d, lag=lag)
    if not np.isfinite(var_d) or var_d <= 0:
        return DieboldMarianoResult(statistic=float("nan"), p_value=float("nan"), n=n, h=h)

    dm = float(np.mean(d) / math.sqrt(var_d / n))
    p = 2.0 * (1.0 - _normal_cdf(abs(dm)))
    return DieboldMarianoResult(statistic=dm, p_value=float(p), n=n, h=h)

