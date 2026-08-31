"""Regression-accuracy and rank-agreement metrics used throughout the study."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _safe(fn, y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3 or np.std(y[m]) == 0 or np.std(p[m]) == 0:
        return float("nan")
    return float(fn(y[m], p[m])[0])


def plcc(y, p) -> float:
    return _safe(stats.pearsonr, y, p)


def srocc(y, p) -> float:
    return _safe(stats.spearmanr, y, p)


def krocc(y, p) -> float:
    return _safe(stats.kendalltau, y, p)


def all_metrics(y, p) -> dict:
    """PLCC/SROCC/KROCC are reported as magnitudes: a metric such as AGIC or LOE
    is inversely oriented with respect to quality, and the strength of the
    monotonic agreement is what is being compared."""
    return {
        "MAE": mae(y, p),
        "RMSE": rmse(y, p),
        "PLCC": abs(plcc(y, p)),
        "SROCC": abs(srocc(y, p)),
        "KROCC": abs(krocc(y, p)),
        "n": int(len(y)),
    }


def agg(list_of_dicts, keys=("MAE", "RMSE", "PLCC", "SROCC", "KROCC")) -> dict:
    out = {}
    for k in keys:
        v = np.array([d[k] for d in list_of_dicts], dtype=float)
        v = v[np.isfinite(v)]
        out[k] = float(np.mean(v)) if v.size else float("nan")
        out[k + "_std"] = float(np.std(v)) if v.size else float("nan")
    return out
