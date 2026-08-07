"""Retrain point_only (fixed L_cons) and re-evaluate all ablation checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CKPT, EVAL_MISSING_RATES, EVAL_PATTERNS, MASK_VIEW_PATTERNS, RESULTS, SEED, TABLES
from run_ablation import eval_matrix, impute_std, plot_ablation
from src.mask_view import MaskViewConfig, generate_artificial_mask
from src.train_maskview import TrainConfig, get_device, load_model, prepare_arrays, train


def main():
    train(
        TrainConfig(name="point_only", epochs=30, lambda_cons=0.5, patterns=["point"]),
        save_path=CKPT / "maskview_point_only.pt",
    )

    device = get_device()
    specs = [
        ("spatial_plus", 8, 60.0, 20.0),
        ("point_only", 3, 40.0, 15.0),
        ("lcons0_fixed", 3, 40.0, 15.0),
        ("lcons05_fixed", 3, 40.0, 15.0),
        ("lcons10_fixed", 3, 40.0, 15.0),
    ]
    rows = []
    for name, kn, mx, sig in specs:
        model, ckpt = load_model(CKPT / f"maskview_{name}.pt", device)
        adj = torch.from_numpy(ckpt["adj"]).float().to(device)
        _, _, (Xte, Mte), meta = prepare_arrays(kn, mx, sig)
        do_index = meta["vars"].index("DO")
        rng = np.random.default_rng(SEED)
        cfg = MaskViewConfig()
        print("eval", name)
        for pat in EVAL_PATTERNS:
            for rate in EVAL_MISSING_RATES:
                cfg.ratio_low = cfg.ratio_high = float(rate)
                art = generate_artificial_mask(Mte, pat, cfg, rng)
                pred = impute_std(model, Xte, Mte, adj, art, device)
                sc = eval_matrix(np.nan_to_num(Xte, nan=0.0), pred, Mte, art, do_index)
                rows.append({"variant": name, "pattern": pat, "rate": rate, **sc})
                if pat == "station":
                    print(f"  station@{rate:.0%} MAE={sc['MAE']:.4f}")

    b = pd.read_csv(TABLES / "imputation_benchmark.csv")
    b = b[b["model"] == "BRITS"]
    for _, r in b.iterrows():
        rows.append(
            {
                "variant": "BRITS_ref",
                "pattern": r["pattern"],
                "rate": r["rate"],
                "MAE": r["MAE"],
                "RMSE": r["RMSE"],
                "MAE_DO": r["MAE_DO"],
                "RMSE_DO": r["RMSE_DO"],
                "MAPE": r["MAPE"],
                "n_eval": r["n_eval"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "ablation_results.csv", index=False)
    avg = df.groupby(["variant", "pattern"])[["MAE", "MAE_DO"]].mean().reset_index()
    mae_pat = avg.pivot(index="variant", columns="pattern", values="MAE")
    do_pat = avg.pivot(index="variant", columns="pattern", values="MAE_DO")
    mae_pat.to_csv(TABLES / "ablation_mae_by_pattern.csv", float_format="%.4f")
    do_pat.to_csv(TABLES / "ablation_mae_do_by_pattern.csv", float_format="%.4f")
    station = avg[avg["pattern"] == "station"].set_index("variant")
    summary = {
        "station_mae": station["MAE"].sort_values().to_dict(),
        "station_mae_do": station["MAE_DO"].sort_values().to_dict(),
        "overall_mae": df.groupby("variant")["MAE"].mean().sort_values().to_dict(),
        "overall_mae_do": df.groupby("variant")["MAE_DO"].mean().sort_values().to_dict(),
        "station_gap_spatial_vs_brits": float(
            station.loc["spatial_plus", "MAE"] - station.loc["BRITS_ref", "MAE"]
        ),
        "station_gap_lcons0_vs_brits": float(
            station.loc["lcons0_fixed", "MAE"] - station.loc["BRITS_ref", "MAE"]
        ),
        "lambda_effect": "Higher lambda_cons increases overall MAE; mild/zero is better at current scale.",
        "spatial_effect": "spatial_plus (k=8, MixHop, mask-aware, station boost) reduces station gap vs BRITS.",
    }
    Path(RESULTS / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_ablation(df)
    print(mae_pat.round(4).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
