"""
Mask-View augmentation for spatiotemporal water-quality imputation.

Core idea (paper selling point)
-------------------------------
CNEMC automatic stations exhibit *block-structured* missingness (sensor downtime,
station outages), not only i.i.d. MCAR holes. Mask-View trains the imputer on
K complementary artificially masked views of the same window, drawn from a
pattern bank that mimics real failure modes:

  - point        : scattered MCAR
  - block_time   : consecutive temporal gaps (downtime)
  - sensor       : entire variable(s) dropped in a window
  - station      : entire station(s) dropped (spatial hole)
  - mixed        : composition of the above

Losses
------
L_rec  : reconstruct originally-observed cells that were artificially masked
L_cons : consistency across views on cells observed in multiple views
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


PATTERN_BANK = ("point", "block_time", "sensor", "station", "mixed")


@dataclass
class MaskViewConfig:
    n_views: int = 3
    patterns: Sequence[str] = ("point", "block_time", "sensor", "station")
    ratio_low: float = 0.15
    ratio_high: float = 0.35
    block_len_range: tuple[int, int] = (3, 12)  # in time steps
    max_sensors_drop: int = 2
    max_stations_drop: int = 3


def _sample_ratio(cfg: MaskViewConfig, rng: np.random.Generator) -> float:
    return float(rng.uniform(cfg.ratio_low, cfg.ratio_high))


def mask_point(shape, obs_mask: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Return artificial mask: 1 = keep, 0 = hide (only on observed cells)."""
    art = np.ones(shape, dtype=np.float32)
    eligible = obs_mask > 0.5
    n = int(eligible.sum())
    if n == 0:
        return art
    n_hide = max(1, int(n * ratio))
    idx = np.flatnonzero(eligible.ravel())
    choose = rng.choice(idx, size=min(n_hide, len(idx)), replace=False)
    flat = art.ravel()
    flat[choose] = 0.0
    return flat.reshape(shape)


def mask_block_time(shape, obs_mask: np.ndarray, ratio: float, rng: np.random.Generator, cfg: MaskViewConfig) -> np.ndarray:
    T, N, D = shape
    art = np.ones(shape, dtype=np.float32)
    target = max(1, int(obs_mask.sum() * ratio))
    hidden = 0
    tries = 0
    while hidden < target and tries < 50:
        tries += 1
        bl = int(rng.integers(cfg.block_len_range[0], cfg.block_len_range[1] + 1))
        t0 = int(rng.integers(0, max(1, T - bl + 1)))
        # drop a random subset of stations/vars in this block
        js = rng.choice(N, size=max(1, N // 3), replace=False)
        ks = rng.choice(D, size=max(1, D // 2), replace=False)
        slab = art[t0 : t0 + bl][:, js][:, :, ks]
        # careful indexing
        for j in js:
            for k in ks:
                art[t0 : t0 + bl, j, k] = 0.0
        hidden = int(((art == 0) & (obs_mask > 0.5)).sum())
    # never hide originally missing (already 0 effect); force keep unobserved? art can be 0 anywhere;
    # evaluation uses obs & ~art
    return art


def mask_sensor(shape, obs_mask: np.ndarray, ratio: float, rng: np.random.Generator, cfg: MaskViewConfig) -> np.ndarray:
    T, N, D = shape
    art = np.ones(shape, dtype=np.float32)
    n_drop = int(rng.integers(1, cfg.max_sensors_drop + 1))
    ks = rng.choice(D, size=min(n_drop, D), replace=False)
    art[:, :, ks] = 0.0
    # if too aggressive vs ratio, randomly restore some stations
    frac = 1.0 - ratio
    if frac > 0:
        restore_n = int(N * frac)
        if restore_n > 0:
            js = rng.choice(N, size=min(restore_n, N), replace=False)
            art[:, js, :] = 1.0
            art[:, :, ks] = art[:, :, ks]  # sensors still dropped on non-restored
            # re-drop sensors on non-restored stations only
            art = np.ones(shape, dtype=np.float32)
            keep_st = set(js.tolist())
            for j in range(N):
                if j not in keep_st:
                    art[:, j, ks] = 0.0
    return art


def mask_station(shape, obs_mask: np.ndarray, ratio: float, rng: np.random.Generator, cfg: MaskViewConfig) -> np.ndarray:
    T, N, D = shape
    art = np.ones(shape, dtype=np.float32)
    n_drop = max(1, min(cfg.max_stations_drop, max(1, int(round(N * ratio)))))
    js = rng.choice(N, size=min(n_drop, N), replace=False)
    art[:, js, :] = 0.0
    return art


def mask_mixed(shape, obs_mask: np.ndarray, ratio: float, rng: np.random.Generator, cfg: MaskViewConfig) -> np.ndarray:
    arts = [
        mask_point(shape, obs_mask, ratio * 0.5, rng),
        mask_block_time(shape, obs_mask, ratio * 0.5, rng, cfg),
    ]
    art = np.minimum(arts[0], arts[1])
    if rng.random() < 0.5:
        art = np.minimum(art, mask_station(shape, obs_mask, ratio * 0.3, rng, cfg))
    return art


def generate_artificial_mask(
    obs_mask: np.ndarray,
    pattern: str,
    cfg: MaskViewConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """obs_mask: (T,N,D) or (B,T,N,D). Returns art_keep mask same shape (1=visible)."""
    cfg = cfg or MaskViewConfig()
    rng = rng or np.random.default_rng()
    single = obs_mask.ndim == 3
    if single:
        obs_mask = obs_mask[None, ...]
    B = obs_mask.shape[0]
    outs = []
    for b in range(B):
        shape = obs_mask[b].shape
        ratio = _sample_ratio(cfg, rng)
        if pattern == "point":
            art = mask_point(shape, obs_mask[b], ratio, rng)
        elif pattern == "block_time":
            art = mask_block_time(shape, obs_mask[b], ratio, rng, cfg)
        elif pattern == "sensor":
            art = mask_sensor(shape, obs_mask[b], ratio, rng, cfg)
        elif pattern == "station":
            art = mask_station(shape, obs_mask[b], ratio, rng, cfg)
        elif pattern == "mixed":
            art = mask_mixed(shape, obs_mask[b], ratio, rng, cfg)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        # artificial hide only meaningful on observed; keep art=1 on missing (no target)
        art = np.where(obs_mask[b] > 0.5, art, 1.0).astype(np.float32)
        outs.append(art)
    out = np.stack(outs, axis=0)
    return out[0] if single else out


def sample_mask_views(
    obs_mask: np.ndarray,
    cfg: MaskViewConfig | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Sample K artificial keep-masks for Mask-View training."""
    cfg = cfg or MaskViewConfig()
    rng = rng or np.random.default_rng()
    patterns = list(cfg.patterns)
    views = []
    for k in range(cfg.n_views):
        pat = patterns[k % len(patterns)]
        # shuffle pattern choice for diversity
        if rng.random() < 0.5:
            pat = patterns[int(rng.integers(0, len(patterns)))]
        views.append(generate_artificial_mask(obs_mask, pat, cfg, rng))
    return views


def apply_view(x_filled: torch.Tensor, obs_mask: torch.Tensor, art_keep: torch.Tensor):
    """
    x_filled: (B,T,N,D) zeros for missing
    obs_mask: original observations
    art_keep: 1 = still visible in this view
    returns view_input, view_mask, eval_mask (positions to reconstruct)
    """
    view_mask = obs_mask * art_keep
    eval_mask = obs_mask * (1.0 - art_keep)  # originally observed, hidden now
    view_input = x_filled * view_mask
    return view_input, view_mask, eval_mask


def mask_view_losses(
    imputations: list[torch.Tensor],
    targets: torch.Tensor,
    eval_masks: list[torch.Tensor],
    view_masks: list[torch.Tensor],
    lambda_cons: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    imputations[k]: (B,T,N,D) prediction for view k
    targets: ground-truth values (nan-safe via masks); use filled finite targets
    """
    rec_terms = []
    for imp, em in zip(imputations, eval_masks):
        if em.sum() < 1:
            continue
        rec_terms.append(F.l1_loss(imp[em > 0.5], targets[em > 0.5]))
    l_rec = torch.stack(rec_terms).mean() if rec_terms else torch.tensor(0.0, device=targets.device)

    cons_terms = []
    n_views = len(imputations)
    for i in range(n_views):
        for j in range(i + 1, n_views):
            both = view_masks[i] * view_masks[j]
            if both.sum() < 1:
                continue
            cons_terms.append(F.l1_loss(imputations[i][both > 0.5], imputations[j][both > 0.5]))
    l_cons = torch.stack(cons_terms).mean() if cons_terms else torch.tensor(0.0, device=targets.device)

    total = l_rec + lambda_cons * l_cons
    return {"loss": total, "l_rec": l_rec.detach(), "l_cons": l_cons.detach()}


def build_eval_corruptions(
    obs_mask: np.ndarray,
    patterns: Sequence[str],
    rates: Sequence[float],
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    For each (pattern, rate) produce art_keep mask for test evaluation.
    Keys: '{pattern}_r{rate}'
    """
    rng = np.random.default_rng(seed)
    out = {}
    cfg = MaskViewConfig()
    for pat in patterns:
        for rate in rates:
            cfg.ratio_low = cfg.ratio_high = float(rate)
            # batch support
            art = generate_artificial_mask(obs_mask, pat, cfg, rng)
            out[f"{pat}_r{rate:.2f}"] = art
    return out
