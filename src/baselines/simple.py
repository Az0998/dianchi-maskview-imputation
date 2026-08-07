"""Classical imputation baselines on (B,T,N,D) arrays with NaN for missing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def impute_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = x.copy()
    # per-variable global mean from observed
    D = x.shape[-1]
    for d in range(D):
        vals = x[..., d][mask[..., d] > 0.5]
        mu = float(np.nanmean(vals)) if len(vals) else 0.0
        miss = mask[..., d] < 0.5
        tmp = out[..., d]
        tmp[miss] = mu
        out[..., d] = tmp
    return out


def impute_locf(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Last observation carried forward along time, per station-variable."""
    out = x.copy()
    B, T, N, D = x.shape
    for b in range(B):
        for n in range(N):
            for d in range(D):
                series = out[b, :, n, d].copy()
                m = mask[b, :, n, d]
                series[m < 0.5] = np.nan
                s = pd.Series(series).ffill().bfill()
                if s.isna().any():
                    s = s.fillna(0.0)
                out[b, :, n, d] = s.to_numpy(dtype=np.float32)
    return out


def impute_linear(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = x.copy()
    B, T, N, D = x.shape
    for b in range(B):
        for n in range(N):
            for d in range(D):
                series = out[b, :, n, d].copy()
                m = mask[b, :, n, d]
                series[m < 0.5] = np.nan
                s = pd.Series(series).interpolate(limit_direction="both")
                if s.isna().any():
                    s = s.fillna(float(np.nanmean(series)) if np.isfinite(series).any() else 0.0)
                out[b, :, n, d] = s.to_numpy(dtype=np.float32)
    return out
