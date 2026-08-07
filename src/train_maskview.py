"""Train MaskView-ST imputer (supports ablation configs)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CKPT,
    DEVICE,
    EPOCHS,
    GAUSS_SIGMA_KM,
    HIDDEN,
    K_NEIGHBORS,
    LAMBDA_CONS,
    LR,
    MASK_VIEW_PATTERNS,
    MAX_DIST_KM,
    N_HEADS,
    N_LAYERS,
    N_VIEWS,
    SEED,
    VIEW_MASK_RATIO,
)
from src.data import (
    ImputationWindowDataset,
    apply_standardize,
    build_adjacency,
    build_panel,
    make_windows,
    standardize_fit,
)
from src.mask_view import MaskViewConfig, apply_view, mask_view_losses, sample_mask_views
from src.models.st_maskview import MaskViewSTImputer


@dataclass
class TrainConfig:
    name: str = "default"
    epochs: int = EPOCHS
    lambda_cons: float = LAMBDA_CONS
    patterns: Sequence[str] = field(default_factory=lambda: list(MASK_VIEW_PATTERNS))
    n_views: int = N_VIEWS
    ratio_low: float = VIEW_MASK_RATIO[0]
    ratio_high: float = VIEW_MASK_RATIO[1]
    k_neighbors: int = K_NEIGHBORS
    max_dist_km: float = MAX_DIST_KM
    gauss_sigma_km: float = GAUSS_SIGMA_KM
    n_layers: int = N_LAYERS
    hidden: int = HIDDEN
    n_heads: int = N_HEADS
    mask_aware: bool = True
    # Upsample station pattern in the bank (helps station outages)
    station_boost: bool = False


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_arrays(k_neighbors: int = K_NEIGHBORS, max_dist_km: float = MAX_DIST_KM, sigma: float = GAUSS_SIGMA_KM):
    panel = build_panel()
    # rebuild adjacency with requested spatial density
    panel.adj = build_adjacency(panel.dist_km, k=k_neighbors, max_km=max_dist_km, sigma=sigma)
    Xtr, Mtr, _ = make_windows(panel, split="train")
    Xva, Mva, _ = make_windows(panel, split="val")
    Xte, Mte, _ = make_windows(panel, split="test")
    mean, std = standardize_fit(Xtr, Mtr)
    Xtr_s = apply_standardize(Xtr, mean, std)
    Xva_s = apply_standardize(Xva, mean, std)
    Xte_s = apply_standardize(Xte, mean, std)
    for arr, m in [(Xtr_s, Mtr), (Xva_s, Mva), (Xte_s, Mte)]:
        arr[m < 0.5] = np.nan
    meta = {
        "mean": mean,
        "std": std,
        "adj": panel.adj,
        "stations": panel.stations,
        "vars": panel.variables,
        "dist_km": panel.dist_km,
    }
    return (Xtr_s, Mtr), (Xva_s, Mva), (Xte_s, Mte), meta


def _pattern_list(cfg: TrainConfig) -> list[str]:
    pats = list(cfg.patterns)
    if cfg.station_boost and "station" in pats:
        # duplicate station to raise sampling frequency
        pats = pats + ["station", "station", "block_time"]
    return pats


def train(cfg: TrainConfig | None = None, save_path: Path | None = None, epochs: int | None = None):
    cfg = cfg or TrainConfig()
    if epochs is not None:
        cfg.epochs = epochs
    set_seed()
    device = get_device()
    (Xtr, Mtr), (Xva, Mva), (Xte, Mte), meta = prepare_arrays(
        cfg.k_neighbors, cfg.max_dist_km, cfg.gauss_sigma_km
    )
    print(
        f"[{cfg.name}] windows {len(Xtr)}/{len(Xva)}/{len(Xte)}  "
        f"k={cfg.k_neighbors} λ_cons={cfg.lambda_cons} patterns={list(cfg.patterns)}  device={device}"
    )

    ds = ImputationWindowDataset(Xtr, Mtr)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = MaskViewSTImputer(
        n_features=Xtr.shape[-1],
        hidden=cfg.hidden,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        mask_aware=cfg.mask_aware,
    ).to(device)
    adj = torch.from_numpy(meta["adj"]).float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    mv_cfg = MaskViewConfig(
        n_views=cfg.n_views,
        patterns=_pattern_list(cfg),
        ratio_low=cfg.ratio_low,
        ratio_high=cfg.ratio_high,
    )
    rng = np.random.default_rng(SEED)

    best_val = float("inf")
    CKPT.mkdir(parents=True, exist_ok=True)
    save_path = save_path or (CKPT / f"maskview_{cfg.name}.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"{cfg.name} ep{epoch}/{cfg.epochs}", leave=False):
            x = batch["x"].to(device)
            m = batch["m"].to(device)
            target = x.clone()

            art_views = sample_mask_views(m.detach().cpu().numpy(), mv_cfg, rng)
            # Use raw decoder outputs for both losses. Blended `imp` must NOT be used for
            # consistency: on visible cells it equals the input, so L_cons would be ~0.
            preds, eval_masks = [], []
            for art in art_views:
                art_t = torch.from_numpy(art).to(device)
                vin, vmask, emask = apply_view(x, m, art_t)
                pred = model.impute_only(vin, vmask, adj)
                preds.append(pred)
                eval_masks.append(emask)

            # Consistency on originally observed cells (view-invariant reconstruction)
            cons_masks = [m for _ in preds]
            out = mask_view_losses(
                preds, target, eval_masks, cons_masks, lambda_cons=cfg.lambda_cons
            )
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(out["loss"].detach().cpu()))

        val_mae = evaluate_val(model, Xva, Mva, adj, device, mv_cfg)
        print(f"[{cfg.name}] epoch {epoch}: loss={np.mean(losses):.4f}  val_mae={val_mae:.4f}")
        if val_mae < best_val:
            best_val = val_mae
            torch.save(
                {
                    "model": model.state_dict(),
                    "mean": meta["mean"],
                    "std": meta["std"],
                    "adj": meta["adj"],
                    "train_cfg": asdict(cfg),
                    "config": {
                        "hidden": cfg.hidden,
                        "n_layers": cfg.n_layers,
                        "n_heads": cfg.n_heads,
                        "n_features": Xtr.shape[-1],
                        "mask_aware": cfg.mask_aware,
                    },
                },
                save_path,
            )
            print(f"  saved → {save_path}")

    return save_path, meta, (Xte, Mte)


@torch.no_grad()
def evaluate_val(model, Xva, Mva, adj, device, cfg: MaskViewConfig) -> float:
    model.eval()
    ds = ImputationWindowDataset(Xva, Mva)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    rng = np.random.default_rng(0)
    errs = []
    for batch in loader:
        x = batch["x"].to(device)
        m = batch["m"].to(device)
        art = sample_mask_views(m.cpu().numpy(), cfg, rng)[0]
        art_t = torch.from_numpy(art).to(device)
        vin, vmask, emask = apply_view(x, m, art_t)
        pred = model.impute_only(vin, vmask, adj)
        if emask.sum() < 1:
            continue
        errs.append(torch.mean(torch.abs(pred[emask > 0.5] - x[emask > 0.5])).item())
    return float(np.mean(errs)) if errs else float("inf")


def load_model(ckpt_path: Path, device=None):
    device = device or get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MaskViewSTImputer(
        n_features=cfg["n_features"],
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        mask_aware=cfg.get("mask_aware", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


if __name__ == "__main__":
    train(TrainConfig(name="default", epochs=EPOCHS))
