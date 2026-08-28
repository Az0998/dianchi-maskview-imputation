#!/usr/bin/env python
"""Dual-protocol evaluation vs EcoInf 2025 (doi:10.1016/j.ecoinf.2025.103283).

Neighbor paper reconstructs Dianchi under random 20–80% holes and 1–4 week
contiguous gaps. This script scores OUR models under that bank, then contrasts
winner identity with the operational Mask-View bank (frozen 105-grid).

No new architecture. Checkpoints are reused, not retrained.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from config import BATCH_SIZE, CKPT, SEED, SEQ_LEN, TABLES, VAL_END
from src.baselines.simple import impute_linear, impute_locf, impute_mean
from src.data import apply_standardize, build_panel, make_windows, standardize_fit
from src.mask_view import MaskViewConfig, apply_view, generate_artificial_mask
from src.metrics import score_all
from src.train_maskview import get_device, load_model

STEPS_PER_DAY = 24 // 4  # 4-hourly
STEPS_PER_WEEK = 7 * STEPS_PER_DAY  # 42
ECOINF_RANDOM_RATES = (0.20, 0.40, 0.60, 0.80)
ECOINF_WEEKS = (1, 2, 3, 4)


def _eval(y_true, y_pred, obs, art, do_index: int) -> dict:
    em = ((obs > 0.5) & (art < 0.5)).astype(np.float32)
    out = score_all(y_true, y_pred, em)
    do_m = em[..., do_index]
    out["MAE_DO"] = score_all(y_true[..., do_index], y_pred[..., do_index], do_m)["MAE"]
    return out


def mask_week_gaps(obs: np.ndarray, n_weeks: int, test_idx: np.ndarray, rng) -> np.ndarray:
    """Independent contiguous gap on each station–variable during the test span.

    EcoInf-style continuous missingness: Linear can interpolate from both sides
    of a 1–4 week hole on a year-long series. Windowed models cannot.
    """
    t, n, d = obs.shape
    steps = n_weeks * STEPS_PER_WEEK
    art = np.ones((t, n, d), dtype=np.float32)
    t0 = int(test_idx.min())
    t1 = int(test_idx.max())
    max_start = t1 - steps + 1
    if max_start < t0:
        art[t0 : t1 + 1] = 0.0
        return np.where(obs > 0.5, art, 1.0).astype(np.float32)
    for j in range(n):
        for k in range(d):
            start = int(rng.integers(t0, max_start + 1))
            art[start : start + steps, j, k] = 0.0
    return np.where(obs > 0.5, art, 1.0).astype(np.float32)


@torch.no_grad()
def maskview_windows(model, X, M, art, adj, device) -> np.ndarray:
    filled = np.nan_to_num(X, nan=0.0).astype(np.float32)
    preds = []
    for i in range(0, len(X), BATCH_SIZE):
        x = torch.from_numpy(filled[i : i + BATCH_SIZE]).to(device)
        m = torch.from_numpy(M[i : i + BATCH_SIZE].astype(np.float32)).to(device)
        a = torch.from_numpy(art[i : i + BATCH_SIZE].astype(np.float32)).to(device)
        vin, vmask, _ = apply_view(x, m, a)
        pred = model.impute_only(vin, vmask, adj)
        imp = vmask * vin + (1.0 - vmask) * pred
        preds.append(imp.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def maskview_stitch(model, panel_std, obs, art, starts, adj, device) -> np.ndarray:
    t, n, d = panel_std.shape
    acc = np.zeros((t, n, d), dtype=np.float32)
    cnt = np.zeros((t, n, d), dtype=np.float32)
    for s in starts:
        x = panel_std[s : s + SEQ_LEN][None]
        m = obs[s : s + SEQ_LEN][None]
        a = art[s : s + SEQ_LEN][None]
        pred = maskview_windows(model, x, m, a, adj, device)[0]
        acc[s : s + SEQ_LEN] += pred
        cnt[s : s + SEQ_LEN] += 1.0
    cnt = np.maximum(cnt, 1.0)
    return acc / cnt


def frozen_operational_rows() -> list[dict]:
    """Contrast rows from the existing 105-grid (pattern-averaged)."""
    frozen = ROOT.parent / "maskview-aquatic-protocol" / "data" / "frozen" / "lake"
    path = frozen / "table_mae_by_pattern_avg.csv"
    if not path.exists():
        return []
    mae = pd.read_csv(path, index_col=0)
    rows = []
    for pat in ("point", "block_time", "sensor", "station"):
        if pat not in mae.columns:
            continue
        col = mae[pat]
        for model in ("Linear", "LOCF", "Mean", "BRITS", "MaskView-ST"):
            if model not in col.index:
                continue
            rows.append(
                {
                    "protocol": "maskview_operational",
                    "setting": pat,
                    "model": model,
                    "MAE": round(float(col.loc[model]), 4),
                    "source": "frozen_pattern_avg",
                }
            )
    abl_path = frozen / "ablation_mae_by_pattern.csv"
    if abl_path.exists():
        abl = pd.read_csv(abl_path, index_col=0)
        if "spatial_plus_l0" in abl.index:
            for pat in ("point", "block_time", "sensor", "station"):
                if pat in abl.columns:
                    rows.append(
                        {
                            "protocol": "maskview_operational",
                            "setting": pat,
                            "model": "spatial_plus_l0",
                            "MAE": round(float(abl.loc["spatial_plus_l0", pat]), 4),
                            "source": "frozen_ablation",
                        }
                    )
    return rows


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    panel = build_panel()
    Xtr, Mtr, _ = make_windows(panel, split="train")
    Xte, Mte, starts = make_windows(panel, split="test")
    mean, std = standardize_fit(Xtr, Mtr)
    Xte_s = apply_standardize(Xte, mean, std)
    Xte_s[Mte < 0.5] = np.nan
    do_index = panel.variables.index("DO")
    test_idx = np.where(panel.times > VAL_END)[0]
    panel_s = apply_standardize(panel.values, mean, std)
    panel_s[panel.mask < 0.5] = np.nan

    device = get_device()
    ckpts = {
        "MaskView-ST": CKPT / "maskview_st.pt",
        "spatial_plus_l0": CKPT / "maskview_spatial_plus_l0.pt",
    }
    loaded = {}
    for name, path in ckpts.items():
        if not path.exists():
            print("missing", path)
            continue
        try:
            model, ckpt = load_model(path, device)
            adj = torch.from_numpy(ckpt["adj"]).float().to(device)
            loaded[name] = (model, adj)
            print(f"loaded {name} from {path.name}")
        except Exception as exc:
            print(f"skip {name}: {exc}")

    rows: list[dict] = []

    # --- A. EcoInf random rates on the same windows as the 105-grid ---
    cfg = MaskViewConfig()
    for rate in ECOINF_RANDOM_RATES:
        cfg.ratio_low = cfg.ratio_high = float(rate)
        art = generate_artificial_mask(Mte, "point", cfg, rng)
        x_in = Xte_s.copy()
        hide = (Mte > 0.5) & (art < 0.5)
        x_in[hide] = np.nan
        m_in = (~np.isnan(x_in)).astype(np.float32)
        y = np.nan_to_num(Xte_s, nan=0.0)
        for name, fn in (("Mean", impute_mean), ("LOCF", impute_locf), ("Linear", impute_linear)):
            pred = fn(x_in, m_in)
            sc = _eval(y, pred, Mte, art, do_index)
            rows.append(
                {
                    "protocol": "ecoinf_random",
                    "setting": f"random_{int(rate*100)}pct",
                    "model": name,
                    "MAE": round(sc["MAE"], 4),
                    "RMSE": round(sc["RMSE"], 4),
                    "MAE_DO": round(sc["MAE_DO"], 4),
                    "n_eval": sc["n_eval"],
                    "source": "windowed_test",
                }
            )
            print(f"ecoinf random {rate:.0%} {name:8s} MAE={sc['MAE']:.4f}")
        for name, (model, adj) in loaded.items():
            pred = maskview_windows(model, Xte_s, Mte, art, adj, device)
            sc = _eval(y, pred, Mte, art, do_index)
            rows.append(
                {
                    "protocol": "ecoinf_random",
                    "setting": f"random_{int(rate*100)}pct",
                    "model": name,
                    "MAE": round(sc["MAE"], 4),
                    "RMSE": round(sc["RMSE"], 4),
                    "MAE_DO": round(sc["MAE_DO"], 4),
                    "n_eval": sc["n_eval"],
                    "source": "windowed_test",
                }
            )
            print(f"ecoinf random {rate:.0%} {name:16s} MAE={sc['MAE']:.4f}")

    # --- B. EcoInf week gaps on the full panel (test span only) ---
    y_full = np.nan_to_num(panel_s, nan=0.0)
    for weeks in ECOINF_WEEKS:
        art = mask_week_gaps(panel.mask, weeks, test_idx, rng)
        x_in = panel_s.copy()
        hide = (panel.mask > 0.5) & (art < 0.5)
        x_in[hide] = np.nan
        m_in = (~np.isnan(x_in)).astype(np.float32)
        # baselines see the full series (EcoInf tensor analog)
        x_b = x_in[None]
        m_b = m_in[None]
        for name, fn in (("Mean", impute_mean), ("LOCF", impute_locf), ("Linear", impute_linear)):
            pred = fn(x_b, m_b)[0]
            sc = _eval(y_full, pred, panel.mask, art, do_index)
            rows.append(
                {
                    "protocol": "ecoinf_week_gap",
                    "setting": f"{weeks}week",
                    "model": name,
                    "MAE": round(sc["MAE"], 4),
                    "RMSE": round(sc["RMSE"], 4),
                    "MAE_DO": round(sc["MAE_DO"], 4),
                    "n_eval": sc["n_eval"],
                    "source": "full_panel_test_holes",
                }
            )
            print(f"ecoinf {weeks}week {name:8s} MAE={sc['MAE']:.4f} n={sc['n_eval']}")
        for name, (model, adj) in loaded.items():
            pred = maskview_stitch(
                model, panel_s, panel.mask, art, starts, adj, device
            )
            sc = _eval(y_full, pred, panel.mask, art, do_index)
            rows.append(
                {
                    "protocol": "ecoinf_week_gap",
                    "setting": f"{weeks}week",
                    "model": name,
                    "MAE": round(sc["MAE"], 4),
                    "RMSE": round(sc["RMSE"], 4),
                    "MAE_DO": round(sc["MAE_DO"], 4),
                    "n_eval": sc["n_eval"],
                    "source": "window_stitch_6day",
                }
            )
            print(f"ecoinf {weeks}week {name:16s} MAE={sc['MAE']:.4f}")

    rows.extend(frozen_operational_rows())
    df = pd.DataFrame(rows)
    out_csv = TABLES / "ecoinf_dual_protocol.csv"
    df.to_csv(out_csv, index=False)

    def winner(sub: pd.DataFrame) -> str:
        scored = sub.dropna(subset=["MAE"])
        if scored.empty:
            return ""
        return str(scored.loc[scored.MAE.idxmin(), "model"])

    lines = [
        "# Dual protocol: EcoInf 2025 missingness vs Mask-View operational bank",
        "",
        "Same Dianchi panel. EcoInf neighbor: random 20-80% and 1-4 week gaps",
        "(doi:10.1016/j.ecoinf.2025.103283). Mask-View rows are frozen pattern averages.",
        "Week-gap Linear/LOCF see the full series; learned model is stitched 6-day windows.",
        "",
        "| Protocol | Setting | Linear | LOCF | MaskView-ST | spatial_plus_l0 | Winner |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    order = (
        [("ecoinf_random", f"random_{p}pct") for p in (20, 40, 60, 80)]
        + [("ecoinf_week_gap", f"{w}week") for w in ECOINF_WEEKS]
        + [("maskview_operational", p) for p in ("point", "block_time", "sensor", "station")]
    )
    for proto, setting in order:
        sub = df[(df.protocol == proto) & (df.setting == setting)]
        if sub.empty:
            continue

        def g(m: str) -> str:
            hit = sub[sub.model == m]
            return f"{float(hit.MAE.iloc[0]):.3f}" if len(hit) else "-"

        lines.append(
            f"| {proto} | {setting} | {g('Linear')} | {g('LOCF')} | "
            f"{g('MaskView-ST')} | {g('spatial_plus_l0')} | {winner(sub)} |"
        )
    lines += [
        "",
        "If Linear wins every EcoInf setting and loses sensor/station, the neighbor",
        "paper's missingness bank never left the interpolable regime.",
        "",
    ]
    out_md = TABLES / "ecoinf_dual_protocol.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", out_csv)
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
