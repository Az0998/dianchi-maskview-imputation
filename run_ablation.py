"""
Ablations + spatial upgrade for MaskView-ST.

Variants
--------
1) λ_cons = 0          (no consistency)
2) λ_cons = 0.5        (default)
3) λ_cons = 1.0
4) patterns = point only
5) spatial+            denser kNN + MixHop/mask-aware + station-pattern boost
6) spatial+ λ=0.5      (main upgraded model)

Evaluates full pattern × rate grid; compares station MAE vs BRITS reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    BATCH_SIZE,
    CKPT,
    EVAL_MISSING_RATES,
    EVAL_PATTERNS,
    FIGS,
    MASK_VIEW_PATTERNS,
    RESULTS,
    SEED,
    TABLES,
)
from src.mask_view import MaskViewConfig, apply_view, generate_artificial_mask  # noqa: E402
from src.metrics import score_all  # noqa: E402
from src.train_maskview import TrainConfig, get_device, load_model, prepare_arrays, train  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")


def eval_matrix(y_true, y_pred, obs_mask, art_keep, do_index: int):
    eval_mask = ((obs_mask > 0.5) & (art_keep < 0.5)).astype(np.float32)
    out = score_all(y_true, y_pred, eval_mask)
    do_m = eval_mask[..., do_index]
    out["MAE_DO"] = score_all(y_true[..., do_index], y_pred[..., do_index], do_m)["MAE"]
    out["RMSE_DO"] = score_all(y_true[..., do_index], y_pred[..., do_index], do_m)["RMSE"]
    return out


@torch.no_grad()
def impute_std(model, Xte, Mte, adj, art, device):
    model.eval()
    filled = np.nan_to_num(Xte, nan=0.0).astype(np.float32)
    preds = []
    for i in range(0, len(Xte), BATCH_SIZE):
        x = torch.from_numpy(filled[i : i + BATCH_SIZE]).to(device)
        m = torch.from_numpy(Mte[i : i + BATCH_SIZE].astype(np.float32)).to(device)
        a = torch.from_numpy(art[i : i + BATCH_SIZE].astype(np.float32)).to(device)
        vin, vmask, _ = apply_view(x, m, a)
        pred = model.impute_only(vin, vmask, adj)
        imp = vmask * vin + (1.0 - vmask) * pred
        preds.append(imp.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def variant_configs(epochs: int) -> list[TrainConfig]:
    full = list(MASK_VIEW_PATTERNS)
    return [
        TrainConfig(name="lcons0", epochs=epochs, lambda_cons=0.0, patterns=full),
        TrainConfig(name="lcons05", epochs=epochs, lambda_cons=0.5, patterns=full),
        TrainConfig(name="lcons10", epochs=epochs, lambda_cons=1.0, patterns=full),
        TrainConfig(name="point_only", epochs=epochs, lambda_cons=0.5, patterns=["point"]),
        TrainConfig(
            name="spatial_plus",
            epochs=epochs,
            lambda_cons=0.5,
            patterns=full,
            k_neighbors=6,
            max_dist_km=55.0,
            gauss_sigma_km=18.0,
            n_layers=3,
            station_boost=True,
            mask_aware=True,
        ),
    ]


def plot_ablation(df: pd.DataFrame):
    FIGS.mkdir(parents=True, exist_ok=True)
    # λ_cons on overall / station
    sub = df[df["variant"].isin(["lcons0", "lcons05", "lcons10"])]
    g = sub.groupby(["variant", "pattern"])["MAE"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.barplot(data=g, x="pattern", y="MAE", hue="variant", ax=ax)
    ax.set_title("λ_cons ablation (MAE by pattern)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ablation_lambda_cons.png", dpi=220)
    plt.close(fig)

    # point_only vs full vs spatial on station
    sub2 = df[df["variant"].isin(["point_only", "lcons05", "spatial_plus"])]
    g2 = sub2.groupby(["variant", "pattern"])["MAE"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.barplot(data=g2, x="pattern", y="MAE", hue="variant", ax=ax)
    ax.set_title("Pattern-bank & spatial upgrade (MAE by pattern)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ablation_pattern_spatial.png", dpi=220)
    plt.close(fig)

    # station gap vs BRITS
    st = df[df["pattern"] == "station"].groupby("variant")[["MAE", "MAE_DO"]].mean()
    if "BRITS_ref" in df["variant"].values:
        pass
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    st2 = st.reset_index().melt(id_vars="variant", value_vars=["MAE", "MAE_DO"])
    sns.barplot(data=st2, x="variant", y="value", hue="variable", ax=ax)
    ax.set_title("Station-pattern skill (lower better)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ablation_station.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    for p in [RESULTS, FIGS, TABLES, CKPT]:
        p.mkdir(parents=True, exist_ok=True)

    device = get_device()
    configs = variant_configs(args.epochs)
    rows = []

    # shared corruptions from a reference panel (k=3); spatial_plus uses its own adj at impute
    _, _, (Xte_ref, Mte_ref), meta_ref = prepare_arrays(k_neighbors=3)
    do_index = meta_ref["vars"].index("DO")
    rng = np.random.default_rng(SEED)
    corruptions = {}
    cfg_m = MaskViewConfig()
    for pat in EVAL_PATTERNS:
        for rate in EVAL_MISSING_RATES:
            cfg_m.ratio_low = cfg_m.ratio_high = float(rate)
            corruptions[(pat, rate)] = generate_artificial_mask(Mte_ref, pat, cfg_m, rng)

    for tcfg in configs:
        ckpt = CKPT / f"maskview_{tcfg.name}.pt"
        if not args.skip_train or not ckpt.exists():
            train(tcfg, save_path=ckpt)
        model, ckpt_obj = load_model(ckpt, device)
        adj = torch.from_numpy(ckpt_obj["adj"]).float().to(device)
        # test arrays must match adjacency/stations from that training prep
        _, _, (Xte, Mte), _ = prepare_arrays(
            tcfg.k_neighbors, tcfg.max_dist_km, tcfg.gauss_sigma_km
        )
        # rebuild corruptions on this Mte with same seed schedule
        rng_i = np.random.default_rng(SEED)
        local_corr = {}
        for pat in EVAL_PATTERNS:
            for rate in EVAL_MISSING_RATES:
                cfg_m.ratio_low = cfg_m.ratio_high = float(rate)
                local_corr[(pat, rate)] = generate_artificial_mask(Mte, pat, cfg_m, rng_i)

        print(f"Evaluating {tcfg.name}…")
        for (pat, rate), art in local_corr.items():
            pred = impute_std(model, Xte, Mte, adj, art, device)
            sc = eval_matrix(np.nan_to_num(Xte, nan=0.0), pred, Mte, art, do_index)
            rows.append({"variant": tcfg.name, "pattern": pat, "rate": rate, **sc})
            if pat == "station":
                print(f"  station@{rate:.0%} MAE={sc['MAE']:.4f} DO={sc['MAE_DO']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "ablation_results.csv", index=False)

    # attach BRITS reference from main benchmark if available
    bench = TABLES / "imputation_benchmark.csv"
    brits_rows = []
    if bench.exists():
        b = pd.read_csv(bench)
        b = b[b["model"] == "BRITS"].copy()
        for _, r in b.iterrows():
            brits_rows.append(
                {
                    "variant": "BRITS_ref",
                    "pattern": r["pattern"],
                    "rate": r["rate"],
                    "MAE": r["MAE"],
                    "RMSE": r.get("RMSE", np.nan),
                    "MAE_DO": r.get("MAE_DO", np.nan),
                    "RMSE_DO": r.get("RMSE_DO", np.nan),
                    "MAPE": r.get("MAPE", np.nan),
                    "n_eval": r.get("n_eval", 0),
                }
            )
        df = pd.concat([df, pd.DataFrame(brits_rows)], ignore_index=True)
        df.to_csv(TABLES / "ablation_results.csv", index=False)

    # pivots
    avg = df.groupby(["variant", "pattern"])[["MAE", "MAE_DO"]].mean().reset_index()
    mae_pat = avg.pivot(index="variant", columns="pattern", values="MAE")
    do_pat = avg.pivot(index="variant", columns="pattern", values="MAE_DO")
    mae_pat.to_csv(TABLES / "ablation_mae_by_pattern.csv", float_format="%.4f")
    do_pat.to_csv(TABLES / "ablation_mae_do_by_pattern.csv", float_format="%.4f")

    station = avg[avg["pattern"] == "station"].set_index("variant")
    summary = {
        "station_mae": station["MAE"].sort_values().to_dict() if len(station) else {},
        "station_mae_do": station["MAE_DO"].sort_values().to_dict() if len(station) else {},
        "overall_mae": df.groupby("variant")["MAE"].mean().sort_values().to_dict(),
        "overall_mae_do": df.groupby("variant")["MAE_DO"].mean().sort_values().to_dict(),
    }
    # gap to BRITS on station
    if "BRITS_ref" in station.index and "spatial_plus" in station.index:
        summary["station_mae_gap_spatial_vs_brits"] = float(
            station.loc["spatial_plus", "MAE"] - station.loc["BRITS_ref", "MAE"]
        )
        summary["station_mae_gap_lcons05_vs_brits"] = float(
            station.loc["lcons05", "MAE"] - station.loc["BRITS_ref", "MAE"]
        ) if "lcons05" in station.index else None
    with open(RESULTS / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_ablation(df)
    print(mae_pat.round(4).to_string())
    print(json.dumps(summary, indent=2))
    print("Wrote ablation tables/figures.")


if __name__ == "__main__":
    main()
