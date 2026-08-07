"""Imputation metrics on artificially masked positions."""
from __future__ import annotations

import numpy as np


def _gather(y_true, y_pred, eval_mask):
    m = eval_mask > 0.5
    yt = y_true[m]
    yp = y_pred[m]
    finite = np.isfinite(yt) & np.isfinite(yp)
    return yt[finite], yp[finite]


def mae(y_true, y_pred, eval_mask) -> float:
    yt, yp = _gather(y_true, y_pred, eval_mask)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred, eval_mask) -> float:
    yt, yp = _gather(y_true, y_pred, eval_mask)
    if len(yt) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred, eval_mask, eps: float = 1e-3) -> float:
    yt, yp = _gather(y_true, y_pred, eval_mask)
    if len(yt) == 0:
        return float("nan")
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs(yt - yp) / denom))


def score_all(y_true, y_pred, eval_mask) -> dict:
    return {
        "MAE": mae(y_true, y_pred, eval_mask),
        "RMSE": rmse(y_true, y_pred, eval_mask),
        "MAPE": mape(y_true, y_pred, eval_mask),
        "n_eval": int((eval_mask > 0.5).sum()),
    }
