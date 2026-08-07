"""
Full imputation benchmark:
  Mean / LOCF / Linear
  SAITS / BRITS / CSDI / StemGNN(GRIN-proxy)
  MaskView-ST (proposed)

Evaluates under Mask-View pattern bank × missing rates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    BATCH_SIZE,
    CKPT,
    EVAL_MISSING_RATES,
    EVAL_PATTERNS,
    FIGS,
    PYPOTS_EPOCHS,
    PYPOTS_PATIENCE,
    RESULTS,
    SEED,
    TABLES,
)
from src.baselines.pypots_models import fit_pypots_model, impute_pypots  # noqa: E402
from src.baselines.simple import impute_linear, impute_locf, impute_mean  # noqa: E402
from src.data import flatten_features  # noqa: E402
from src.mask_view import MaskViewConfig, apply_view, generate_artificial_mask  # noqa: E402
from src.metrics import score_all  # noqa: E402
from src.train_maskview import get_device, load_model, prepare_arrays, train as train_maskview  # noqa: E402
from src.viz import (  # noqa: E402
    plot_benchmark_heatmap,
    plot_imputation_example,
    plot_mask_view_schematic,
    plot_model_bars,
    plot_pattern_radar_or_lines,
)


def invert_standardize(x, mean, std):
    return x * std + mean


@torch.no_grad()
def impute_maskview(model, X, M, adj, art_keep, device, mean, std):
    """X has NaN for missing; returns imputed in original scale (B,T,N,D)."""
    filled = np.nan_to_num(X, nan=0.0).astype(np.float32)
    ds_x = torch.from_numpy(filled)
    ds_m = torch.from_numpy(M.astype(np.float32))
    art = torch.from_numpy(art_keep.astype(np.float32))
    preds = []
    for i in range(0, len(X), BATCH_SIZE):
        x = ds_x[i : i + BATCH_SIZE].to(device)
        m = ds_m[i : i + BATCH_SIZE].to(device)
        a = art[i : i + BATCH_SIZE].to(device)
        vin, vmask, _ = apply_view(x, m, a)
        pred = model.impute_only(vin, vmask, adj)
        imp = vmask * vin + (1.0 - vmask) * pred
        preds.append(imp.cpu().numpy())
    out = np.concatenate(preds, axis=0)
    return invert_standardize(out, mean, std)


def eval_matrix(y_true, y_pred, obs_mask, art_keep, do_index: int | None = None):
    """Primary metrics in the provided scale (prefer standardized for fairness)."""
    eval_mask = ((obs_mask > 0.5) & (art_keep < 0.5)).astype(np.float32)
    out = score_all(y_true, y_pred, eval_mask)
    if do_index is not None:
        do_m = eval_mask[..., do_index]
        out["MAE_DO"] = score_all(y_true[..., do_index], y_pred[..., do_index], do_m)["MAE"]
        out["RMSE_DO"] = score_all(y_true[..., do_index], y_pred[..., do_index], do_m)["RMSE"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="MaskView epochs")
    parser.add_argument("--pypots-epochs", type=int, default=PYPOTS_EPOCHS)
    parser.add_argument("--skip-pypots", action="store_true", help="Only classical + MaskView")
    parser.add_argument("--skip-csdi", action="store_true", default=True, help="Skip heavy CSDI by default")
    parser.add_argument("--include-csdi", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Fewer rates/patterns for smoke test")
    args = parser.parse_args()
    if args.include_csdi:
        args.skip_csdi = False

    for p in [RESULTS, FIGS, TABLES, CKPT]:
        p.mkdir(parents=True, exist_ok=True)

    plot_mask_view_schematic(FIGS / "fig_maskview_patterns.png")

    print("Preparing arrays…")
    (Xtr, Mtr), (Xva, Mva), (Xte, Mte), meta = prepare_arrays()
    mean, std = meta["mean"], meta["std"]
    # Standardized tensors for fair multi-var comparison (NaN on missing)
    Xte_std = Xte.copy()
    Xtr_std = Xtr.copy()
    do_index = meta["vars"].index("DO")

    device = get_device()
    ckpt = CKPT / "maskview_st.pt"
    if (not ckpt.exists()) or args.epochs:
        print("Training MaskView-ST…")
        # remove stale ckpt if re-train requested
        if args.epochs and ckpt.exists():
            ckpt.unlink()
        train_maskview(epochs=args.epochs or 20, save_path=ckpt)
    else:
        print(f"Using existing checkpoint {ckpt}")

    model, ckpt_obj = load_model(ckpt, device)
    adj = torch.from_numpy(ckpt_obj["adj"]).float().to(device)

    patterns = ["point", "block_time", "mixed"] if args.quick else list(EVAL_PATTERNS)
    rates = [0.2] if args.quick else list(EVAL_MISSING_RATES)

    rows = []
    rng = np.random.default_rng(SEED)

    # Prebuild corruptions for test
    corruptions = {}
    cfg = MaskViewConfig()
    for pat in patterns:
        for rate in rates:
            cfg.ratio_low = cfg.ratio_high = float(rate)
            art = generate_artificial_mask(Mte, pat, cfg, rng)
            corruptions[(pat, rate)] = art

    @torch.no_grad()
    def maskview_std(art):
        """Impute in standardized space (no invert)."""
        model.eval()
        filled = np.nan_to_num(Xte_std, nan=0.0).astype(np.float32)
        preds = []
        for i in range(0, len(Xte_std), BATCH_SIZE):
            x = torch.from_numpy(filled[i : i + BATCH_SIZE]).to(device)
            m = torch.from_numpy(Mte[i : i + BATCH_SIZE].astype(np.float32)).to(device)
            a = torch.from_numpy(art[i : i + BATCH_SIZE].astype(np.float32)).to(device)
            vin, vmask, _ = apply_view(x, m, a)
            pred = model.impute_only(vin, vmask, adj)
            imp = vmask * vin + (1.0 - vmask) * pred
            preds.append(imp.detach().cpu().numpy())
        return np.concatenate(preds, axis=0)

    # ---- classical (standardized space) ----
    for name, fn in [("Mean", impute_mean), ("LOCF", impute_locf), ("Linear", impute_linear)]:
        print(f"Evaluating {name}…")
        for (pat, rate), art in corruptions.items():
            x_in = Xte_std.copy()
            hide = (Mte > 0.5) & (art < 0.5)
            x_in[hide] = np.nan
            m_in = (~np.isnan(x_in)).astype(np.float32)
            pred = fn(x_in, m_in)
            sc = eval_matrix(np.nan_to_num(Xte_std, nan=0.0), pred, Mte, art, do_index)
            rows.append({"model": name, "pattern": pat, "rate": rate, **sc})

    # ---- MaskView ----
    print("Evaluating MaskView-ST…")
    for (pat, rate), art in corruptions.items():
        pred = maskview_std(art)
        sc = eval_matrix(np.nan_to_num(Xte_std, nan=0.0), pred, Mte, art, do_index)
        rows.append({"model": "MaskView-ST", "pattern": pat, "rate": rate, **sc})

    # ---- PyPOTS (fit once per model, evaluate all corruptions) ----
    if not args.skip_pypots:
        def prep_flat(X, M, art=None):
            Xc = X.copy()
            Mc = M.copy()
            if art is not None:
                hide = (M > 0.5) & (art < 0.5)
                Xc = Xc.copy()
                Xc[hide] = np.nan
                Mc = (~np.isnan(Xc)).astype(np.float32)
            Xf = np.nan_to_num(Xc, nan=0.0)
            return flatten_features(Xf), flatten_features(Mc)

        pypots_names = ["SAITS", "BRITS", "STEMGNN"]
        if not args.skip_csdi:
            pypots_names.append("CSDI")

        tr_x, tr_m = prep_flat(Xtr, Mtr)
        # paper run: use all windows; quick-ish cap only if enormous
        max_tr = len(tr_x) if not args.quick else min(256, len(tr_x))
        max_te = len(Xte) if not args.quick else min(128, len(Xte))
        tr_x, tr_m = tr_x[:max_tr], tr_m[:max_tr]
        n, d = Xte.shape[2], Xte.shape[3]

        for name in pypots_names:
            print(f"Fitting PyPOTS {name} once…")
            label = "GRIN*" if name == "STEMGNN" else name
            try:
                model_p = fit_pypots_model(
                    name,
                    tr_x,
                    tr_m,
                    epochs=args.pypots_epochs,
                    patience=PYPOTS_PATIENCE,
                    device=str(device),
                )
                for (pat, rate), art in corruptions.items():
                    te_x, te_m = prep_flat(Xte[:max_te], Mte[:max_te], art[:max_te])
                    imp = impute_pypots(model_p, te_x, te_m)
                    imp_std = imp.reshape(imp.shape[0], imp.shape[1], n, d)
                    sc = eval_matrix(
                        np.nan_to_num(Xte_std[:max_te], nan=0.0),
                        imp_std,
                        Mte[:max_te],
                        art[:max_te],
                        do_index,
                    )
                    rows.append({"model": label, "pattern": pat, "rate": rate, **sc})
                    print(f"  {label} | {pat}@{rate:.0%} MAE={sc['MAE']:.4f} DO={sc.get('MAE_DO', float('nan')):.4f}")
            except Exception as e:
                print(f"  !! {name} failed: {e}")
                rows.append(
                    {
                        "model": label,
                        "pattern": "error",
                        "rate": 0.0,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "MAPE": np.nan,
                        "n_eval": 0,
                        "error": str(e),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "imputation_benchmark.csv", index=False)
    print(df.to_string(index=False))

    # plots
    df_ok = df[df["pattern"] != "error"].copy()
    if len(df_ok):
        plot_benchmark_heatmap(df_ok, "MAE", FIGS / "fig_imputation_mae_heatmap.png")
        plot_benchmark_heatmap(df_ok, "RMSE", FIGS / "fig_imputation_rmse_heatmap.png")
        if (df_ok["pattern"] == "mixed").any():
            rate_plot = 0.2 if 0.2 in df_ok["rate"].values else df_ok["rate"].iloc[0]
            plot_model_bars(df_ok, "MAE", FIGS / "fig_imputation_mae_bars_mixed.png", "mixed", rate_plot)
        if (df_ok["model"] == "MaskView-ST").any():
            plot_pattern_radar_or_lines(df_ok, "MaskView-ST", "MAE", FIGS / "fig_maskview_mae_by_pattern.png")

    # example visualization on one window (original scale)
    art = corruptions[list(corruptions.keys())[0]]
    pred_std = maskview_std(art)[:1]
    pred_orig = invert_standardize(pred_std, mean, std)
    obs_orig = invert_standardize(np.nan_to_num(Xte[:1], nan=0.0), mean, std)
    t = np.arange(pred_orig.shape[1])
    hide = (Mte[0, :, 0, do_index] > 0.5) & (art[0, :, 0, do_index] < 0.5)
    plot_imputation_example(
        t,
        obs_orig[0, :, 0, do_index],
        pred_orig[0, :, 0, do_index],
        hide.astype(np.float32),
        "DO",
        meta["stations"][0],
        FIGS / "fig_imputation_example_DO.png",
    )

    summary = {
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "patterns": patterns,
        "rates": rates,
        "best_by_mae": df_ok.sort_values("MAE").groupby("model")["MAE"].mean().sort_values().to_dict()
        if len(df_ok)
        else {},
    }
    with open(RESULTS / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Done.", RESULTS)


if __name__ == "__main__":
    main()
