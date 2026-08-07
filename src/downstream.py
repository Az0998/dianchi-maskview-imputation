"""
Downstream tasks after imputation (插补是手段不是目的):
  1) Water-quality anomaly warning — DO sudden drop / TN exceedance
  2) Algal-bloom proxy — high NTU / nutrient surge (chl-a not in release; use proxies)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DO_DROP_THR, TN_EXCEED, VARS


def do_index(variables: list[str] | None = None) -> int:
    variables = variables or VARS
    return variables.index("DO")


def tn_index(variables: list[str] | None = None) -> int:
    variables = variables or VARS
    return variables.index("TN")


def detect_do_drop(do_series: np.ndarray, steps_24h: int = 6, thr: float = DO_DROP_THR) -> np.ndarray:
    """Binary labels: DO drops by >= thr within next 24h window."""
    T = len(do_series)
    y = np.zeros(T, dtype=np.int32)
    for t in range(T - steps_24h):
        if not np.isfinite(do_series[t]):
            continue
        future = do_series[t : t + steps_24h + 1]
        if np.nanmin(future) <= do_series[t] - thr:
            y[t] = 1
    return y


def detect_tn_exceed(tn_series: np.ndarray, thr: float = TN_EXCEED) -> np.ndarray:
    return (np.nan_to_num(tn_series, nan=-np.inf) >= thr).astype(np.int32)


def bloom_proxy_score(ntu, tn, tp) -> np.ndarray:
    """Simple z-score sum proxy when chl-a is unavailable."""
    def z(a):
        m, s = np.nanmean(a), np.nanstd(a)
        s = s if s > 1e-6 else 1.0
        return (a - m) / s

    return z(ntu) + 0.5 * z(tn) + 0.5 * z(tp)


def evaluate_warning(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall == precision + recall and (precision + recall) else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compare_raw_vs_imputed_warning(
    raw_do: np.ndarray,
    imp_do: np.ndarray,
    steps_24h: int = 6,
) -> pd.DataFrame:
    """Show that imputation recovers more usable warning labels / continuity."""
    rows = []
    for name, series in [("raw", raw_do), ("imputed", imp_do)]:
        labels = detect_do_drop(series, steps_24h=steps_24h)
        rows.append(
            {
                "series": name,
                "valid_ratio": float(np.isfinite(series).mean()),
                "event_rate": float(labels.mean()),
                "n_events": int(labels.sum()),
            }
        )
    return pd.DataFrame(rows)
