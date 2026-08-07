"""Re-run λ_cons ablation after consistency-loss bugfix."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CKPT, EVAL_MISSING_RATES, EVAL_PATTERNS, MASK_VIEW_PATTERNS, RESULTS, SEED, TABLES  # noqa: E402
from run_ablation import eval_matrix, impute_std  # noqa: E402
from src.mask_view import MaskViewConfig, generate_artificial_mask  # noqa: E402
from src.train_maskview import TrainConfig, get_device, load_model, prepare_arrays, train  # noqa: E402


def main(epochs: int = 30):
    device = get_device()
    variants = [
        TrainConfig(name="lcons0_fixed", epochs=epochs, lambda_cons=0.0, patterns=list(MASK_VIEW_PATTERNS)),
        TrainConfig(name="lcons05_fixed", epochs=epochs, lambda_cons=0.5, patterns=list(MASK_VIEW_PATTERNS)),
        TrainConfig(name="lcons10_fixed", epochs=epochs, lambda_cons=1.0, patterns=list(MASK_VIEW_PATTERNS)),
    ]
    rows = []
    do_index = None
    for tcfg in variants:
        ckpt = CKPT / f"maskview_{tcfg.name}.pt"
        train(tcfg, save_path=ckpt)
        model, ckpt_obj = load_model(ckpt, device)
        adj = __import__("torch").from_numpy(ckpt_obj["adj"]).float().to(device)
        _, _, (Xte, Mte), meta = prepare_arrays(tcfg.k_neighbors, tcfg.max_dist_km, tcfg.gauss_sigma_km)
        do_index = meta["vars"].index("DO")
        rng = __import__("numpy").random.default_rng(SEED)
        cfg_m = MaskViewConfig()
        print(f"Evaluating {tcfg.name}…")
        for pat in EVAL_PATTERNS:
            for rate in EVAL_MISSING_RATES:
                cfg_m.ratio_low = cfg_m.ratio_high = float(rate)
                art = generate_artificial_mask(Mte, pat, cfg_m, rng)
                pred = impute_std(model, Xte, Mte, adj, art, device)
                sc = eval_matrix(__import__("numpy").nan_to_num(Xte, nan=0.0), pred, Mte, art, do_index)
                rows.append({"variant": tcfg.name, "pattern": pat, "rate": rate, **sc})

    # merge with previous non-lambda rows
    prev = TABLES / "ablation_results.csv"
    df_new = pd.DataFrame(rows)
    if prev.exists():
        old = pd.read_csv(prev)
        keep = old[~old["variant"].isin(["lcons0", "lcons05", "lcons10"])]
        # drop old fixed if re-run
        keep = keep[~keep["variant"].str.endswith("_fixed")]
        df = pd.concat([keep, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(TABLES / "ablation_results.csv", index=False)

    avg = df.groupby(["variant", "pattern"])[["MAE", "MAE_DO"]].mean().reset_index()
    mae_pat = avg.pivot(index="variant", columns="pattern", values="MAE")
    mae_pat.to_csv(TABLES / "ablation_mae_by_pattern.csv", float_format="%.4f")
    print(mae_pat.round(4).to_string())

    # update summary for fixed lambdas + spatial
    station = avg[avg["pattern"] == "station"].set_index("variant")["MAE"].sort_values()
    summary = {
        "note": "lcons*_fixed uses consistency on raw predictions over originally observed cells",
        "station_mae": station.to_dict(),
        "overall_mae": df.groupby("variant")["MAE"].mean().sort_values().to_dict(),
        "lambda_fixed_overall": df_new.groupby("variant")["MAE"].mean().sort_values().to_dict(),
        "lambda_fixed_station": df_new[df_new.pattern == "station"].groupby("variant")["MAE"].mean().sort_values().to_dict(),
    }
    with open(RESULTS / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(30)
