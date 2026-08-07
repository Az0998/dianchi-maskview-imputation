"""Run downstream anomaly-warning comparison: raw vs MaskView-imputed series."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CKPT, FIGS, RESULTS, TABLES  # noqa: E402
from src.downstream import compare_raw_vs_imputed_warning, detect_tn_exceed  # noqa: E402
from src.train_maskview import get_device, load_model, prepare_arrays  # noqa: E402
from run_benchmark import impute_maskview, invert_standardize  # noqa: E402


def main():
    for p in [RESULTS, FIGS, TABLES]:
        p.mkdir(parents=True, exist_ok=True)

    _, _, (Xte, Mte), meta = prepare_arrays()
    mean, std = meta["mean"], meta["std"]
    device = get_device()
    ckpt = CKPT / "maskview_st.pt"
    if not ckpt.exists():
        raise SystemExit("Train MaskView first: python -m src.train_maskview  or run_benchmark.py")

    model, ckpt_obj = load_model(ckpt, device)
    adj = torch.from_numpy(ckpt_obj["adj"]).float().to(device)

    art = np.ones_like(Mte)
    pred = impute_maskview(model, Xte, Mte, adj, art, device, mean, std)
    raw = invert_standardize(np.nan_to_num(Xte, nan=0.0), mean, std).copy()
    raw[Mte < 0.5] = np.nan

    do_i = meta["vars"].index("DO")
    tn_i = meta["vars"].index("TN")

    do_rows, tn_rows = [], []
    for j, st in enumerate(meta["stations"]):
        raw_do = raw[0, :, j, do_i]
        imp_do = pred[0, :, j, do_i]
        cmp = compare_raw_vs_imputed_warning(raw_do, imp_do)
        cmp["station"] = st
        cmp["task"] = "DO_drop"
        do_rows.append(cmp)

        raw_tn = raw[0, :, j, tn_i]
        imp_tn = pred[0, :, j, tn_i]
        tn_rows.append(
            {
                "series": "raw",
                "station": st,
                "task": "TN_exceed",
                "valid_ratio": float(np.isfinite(raw_tn).mean()),
                "event_rate": float(detect_tn_exceed(raw_tn).mean()),
                "n_events": int(detect_tn_exceed(raw_tn).sum()),
            }
        )
        tn_rows.append(
            {
                "series": "imputed",
                "station": st,
                "task": "TN_exceed",
                "valid_ratio": float(np.isfinite(imp_tn).mean()),
                "event_rate": float(detect_tn_exceed(imp_tn).mean()),
                "n_events": int(detect_tn_exceed(imp_tn).sum()),
            }
        )

    out = pd.concat([pd.concat(do_rows, ignore_index=True), pd.DataFrame(tn_rows)], ignore_index=True)
    out.to_csv(TABLES / "downstream_warning_raw_vs_imputed.csv", index=False)
    print(out.groupby(["task", "series"])[["valid_ratio", "event_rate"]].mean())
    print("Wrote", TABLES / "downstream_warning_raw_vs_imputed.csv")


if __name__ == "__main__":
    main()
