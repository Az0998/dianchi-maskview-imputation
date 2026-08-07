"""Load Dianchi data into spatiotemporal tensors and sliding windows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    DISTANCE_CSV,
    FREQ,
    K_NEIGHBORS,
    MAX_DIST_KM,
    GAUSS_SIGMA_KM,
    PARQUET,
    SEQ_LEN,
    STRIDE,
    TRAIN_END,
    VAL_END,
    VARS,
)


@dataclass
class Panel:
    """Full panel: values/mask shaped (T, N, D), plus metadata."""

    values: np.ndarray  # float32, NaN for missing
    mask: np.ndarray  # 1 = observed
    times: pd.DatetimeIndex
    stations: list[str]
    variables: list[str]
    dist_km: np.ndarray  # (N, N)
    adj: np.ndarray  # (N, N) sparse gaussian weights


def load_raw(path=PARQUET) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["tm"] = pd.to_datetime(df["tm"])
    return df.sort_values(["station", "tm"]).reset_index(drop=True)


def load_distance(path=DISTANCE_CSV) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce")


def build_adjacency(dist: np.ndarray, k: int = K_NEIGHBORS, max_km: float = MAX_DIST_KM, sigma: float = GAUSS_SIGMA_KM):
    n = dist.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        order = np.argsort(dist[i])
        chosen = [j for j in order if j != i and dist[i, j] <= max_km][:k]
        for j in chosen:
            w = np.exp(-(dist[i, j] ** 2) / (2 * sigma**2))
            adj[i, j] = w
        if adj[i].sum() > 0:
            adj[i] /= adj[i].sum()
        adj[i, i] = 1.0
    # symmetrize
    adj = np.maximum(adj, adj.T)
    return adj


def build_panel(df: pd.DataFrame | None = None) -> Panel:
    df = load_raw() if df is None else df
    dist_df = load_distance()
    stations = [s for s in dist_df.index.tolist() if s in set(df["station"])]
    # align distance matrix order
    dist_df = dist_df.loc[stations, stations]
    t_min, t_max = df["tm"].min(), df["tm"].max()
    times = pd.date_range(t_min.floor(FREQ), t_max.ceil(FREQ), freq=FREQ)

    T, N, D = len(times), len(stations), len(VARS)
    values = np.full((T, N, D), np.nan, dtype=np.float32)
    time_to_i = {pd.Timestamp(t): i for i, t in enumerate(times)}
    st_to_j = {s: j for j, s in enumerate(stations)}

    df = df[df["station"].isin(st_to_j)].copy()
    df["_ti"] = df["tm"].map(time_to_i)
    df["_sj"] = df["station"].map(st_to_j)
    df = df.dropna(subset=["_ti", "_sj"])
    ti = df["_ti"].astype(int).to_numpy()
    sj = df["_sj"].astype(int).to_numpy()
    for var_k, var in enumerate(VARS):
        if var not in df.columns:
            continue
        vals = df[var].to_numpy(dtype=np.float32)
        ok = np.isfinite(vals)
        values[ti[ok], sj[ok], var_k] = vals[ok]

    mask = (~np.isnan(values)).astype(np.float32)
    dist = dist_df.values.astype(np.float32)
    adj = build_adjacency(dist)
    return Panel(values, mask, times, stations, list(VARS), dist, adj)


def _split_time_indices(times: pd.DatetimeIndex):
    train_idx = np.where(times <= TRAIN_END)[0]
    val_idx = np.where((times > TRAIN_END) & (times <= VAL_END))[0]
    test_idx = np.where(times > VAL_END)[0]
    return train_idx, val_idx, test_idx


def make_windows(panel: Panel, seq_len: int = SEQ_LEN, stride: int = STRIDE, split: str = "train"):
    train_idx, val_idx, test_idx = _split_time_indices(panel.times)
    if split == "train":
        allowed = set(train_idx.tolist())
        lo, hi = int(train_idx.min()), int(train_idx.max())
    elif split == "val":
        allowed = set(val_idx.tolist())
        lo, hi = int(val_idx.min()), int(val_idx.max())
    else:
        allowed = set(test_idx.tolist())
        lo, hi = int(test_idx.min()), int(test_idx.max())

    starts = [
        s
        for s in range(lo, hi - seq_len + 2, stride)
        if all(i in allowed for i in range(s, s + seq_len))
    ]
    if not starts:
        raise RuntimeError(f"No windows for split={split}")
    X = np.stack([panel.values[s : s + seq_len] for s in starts], axis=0)  # (B,T,N,D)
    M = np.stack([panel.mask[s : s + seq_len] for s in starts], axis=0)
    return X, M, starts


class ImputationWindowDataset(Dataset):
    def __init__(self, values: np.ndarray, mask: np.ndarray, fill: float = 0.0):
        self.values = values.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.filled = np.nan_to_num(self.values, nan=fill)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.filled[idx]),
            "m": torch.from_numpy(self.mask[idx]),
            "y": torch.from_numpy(self.values[idx]),  # with NaN kept as nan → use mask
        }


def flatten_features(x: np.ndarray) -> np.ndarray:
    """(B,T,N,D) -> (B,T,N*D) for PyPOTS."""
    b, t, n, d = x.shape
    return x.reshape(b, t, n * d)


def standardize_fit(x: np.ndarray, m: np.ndarray):
    """Fit mean/std on observed cells; x may contain NaN."""
    obs = m > 0.5
    mean = np.zeros(x.shape[-1], dtype=np.float32) if x.ndim == 3 else None
    # per-variable across stations: shape D
    # x: (B,T,N,D)
    D = x.shape[-1]
    mean = np.zeros(D, dtype=np.float32)
    std = np.ones(D, dtype=np.float32)
    for d in range(D):
        vals = x[..., d][obs[..., d]]
        if len(vals) == 0:
            continue
        mean[d] = float(np.nanmean(vals))
        s = float(np.nanstd(vals))
        std[d] = s if s > 1e-6 else 1.0
    return mean, std


def apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (x - mean) / std
    return out
