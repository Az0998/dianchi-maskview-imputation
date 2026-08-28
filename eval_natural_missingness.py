#!/usr/bin/env python
"""Natural missingness: taxonomy + year-shift replay with ground truth.

Naturally missing cells have no labels. We (1) classify every natural hole on
the 4 h grid, and (2) replay another year's missing calendar onto 2024 cells
that *were* observed — the closest operational hold-out with targets.

MCAR at the same hide rate is the control. Same models as the EcoInf dual
protocol; no new architecture.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from config import BATCH_SIZE, CKPT, SEED, SEQ_LEN, TABLES, TRAIN_END, VAL_END
from src.baselines.simple import impute_linear, impute_locf, impute_mean
from src.data import apply_standardize, build_panel, make_windows, standardize_fit
from src.mask_view import apply_view, mask_point
from src.metrics import score_all
from src.train_maskview import get_device, load_model

STEPS_PER_DAY = 6


def _eval(y_true, y_pred, obs, art, do_index: int) -> dict:
    em = ((obs > 0.5) & (art < 0.5)).astype(np.float32)
    out = score_all(y_true, y_pred, em)
    do_m = em[..., do_index]
    out["MAE_DO"] = score_all(y_true[..., do_index], y_pred[..., do_index], do_m)["MAE"]
    return out


def classify_natural_missing(obs: np.ndarray) -> dict:
    """obs: (T,N,D) 1=observed. Returns counts over missing cells."""
    miss = obs < 0.5
    t, n, d = obs.shape
    station_tn = miss.all(axis=2)
    station = miss & station_tn[:, :, None]
    remain = miss & ~station

    run_len = np.zeros((t, n, d), dtype=np.int32)
    for j in range(n):
        for k in range(d):
            col = remain[:, j, k]
            i = 0
            while i < t:
                if not col[i]:
                    i += 1
                    continue
                i0 = i
                while i < t and col[i]:
                    i += 1
                run_len[i0:i, j, k] = i - i0

    point = remain & (run_len == 1)
    sensor = remain & (run_len >= STEPS_PER_DAY)
    block = remain & ~point & ~sensor
    n_miss = int(miss.sum())
    def frac(x):
        return float(x.sum() / n_miss) if n_miss else 0.0

    station_runs = []
    station_runs_cell = []
    for j in range(n):
        col = station_tn[:, j]
        i = 0
        while i < t:
            if not col[i]:
                i += 1
                continue
            i0 = i
            while i < t and col[i]:
                i += 1
            L = i - i0
            station_runs.append(L)
            station_runs_cell.extend([L] * L)
    st = np.asarray(station_runs, dtype=np.float64) if station_runs else np.array([0.0])
    stc = np.asarray(station_runs_cell, dtype=np.float64) if station_runs_cell else np.array([0.0])
    runs = run_len[remain]
    return {
        "n_cells": int(obs.size),
        "n_missing": n_miss,
        "missing_rate": float(n_miss / obs.size),
        "frac_station": frac(station),
        "frac_sensor_ge1day": frac(sensor),
        "frac_block_2to5steps": frac(block),
        "frac_point_isolated": frac(point),
        "frac_non_point": 1.0 - frac(point) if n_miss else 0.0,
        "run_len_p50": float(np.median(runs)) if runs.size else 0.0,
        "run_len_p90": float(np.percentile(runs, 90)) if runs.size else 0.0,
        "run_len_p99": float(np.percentile(runs, 99)) if runs.size else 0.0,
        "station_outage_time_frac": float(station_tn.mean()),
        "station_outage_run_p50_steps": float(np.median(st)),
        "station_outage_run_p90_steps": float(np.percentile(st, 90)),
        "station_outage_run_p50_days": float(np.median(st) / STEPS_PER_DAY),
        "station_outage_run_p90_days": float(np.percentile(st, 90) / STEPS_PER_DAY),
        "station_outage_cell_p50_days": float(np.median(stc) / STEPS_PER_DAY),
        "station_outage_cell_p90_days": float(np.percentile(stc, 90) / STEPS_PER_DAY),
        "n_station_outage_runs": int(len(station_runs)),
    }


def year_shift_art(times: pd.DatetimeIndex, obs: np.ndarray, src_year: int, dst_year: int) -> np.ndarray:
    """Hide dst cells that were missing in src on the same (month, day, hour)."""
    art = np.ones_like(obs, dtype=np.float32)
    src_idx = np.where(times.year == src_year)[0]
    dst_idx = np.where(times.year == dst_year)[0]
    key_to_src: dict[tuple[int, int, int], int] = {}
    for i in src_idx:
        ts = times[i]
        key_to_src[(int(ts.month), int(ts.day), int(ts.hour))] = int(i)
    for i in dst_idx:
        ts = times[i]
        j = key_to_src.get((int(ts.month), int(ts.day), int(ts.hour)))
        if j is None:
            continue
        src_miss = obs[j] < 0.5
        dst_obs = obs[i] > 0.5
        hide = src_miss & dst_obs
        row = art[i]
        row[hide] = 0.0
        art[i] = row
    return art


def mcar_match_rate(obs: np.ndarray, art_ref: np.ndarray, rng: np.random.Generator, restrict) -> np.ndarray:
    """MCAR hide of originally observed cells inside `restrict`, same count as art_ref."""
    hide_ref = (obs > 0.5) & (art_ref < 0.5) & restrict
    n_hide = int(hide_ref.sum())
    eligible = (obs > 0.5) & restrict
    art = np.ones_like(obs, dtype=np.float32)
    idx = np.flatnonzero(eligible.ravel())
    if n_hide == 0 or idx.size == 0:
        return art
    choose = rng.choice(idx, size=min(n_hide, idx.size), replace=False)
    flat = art.ravel()
    flat[choose] = 0.0
    return flat.reshape(obs.shape)


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
    return acc / np.maximum(cnt, 1.0)


def score_art(name, pred, y, obs, art, do_index, protocol, setting, source) -> dict:
    sc = _eval(y, pred, obs, art, do_index)
    return {
        "protocol": protocol,
        "setting": setting,
        "model": name,
        "MAE": round(sc["MAE"], 4),
        "RMSE": round(sc["RMSE"], 4),
        "MAE_DO": round(sc["MAE_DO"], 4),
        "n_eval": sc["n_eval"],
        "hide_rate_on_obs": round(
            float(((obs > 0.5) & (art < 0.5)).sum() / max((obs > 0.5).sum(), 1)), 4
        ),
        "source": source,
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    panel = build_panel()
    times = pd.DatetimeIndex(panel.times)
    obs = panel.mask
    splits = {
        "all": np.ones(len(times), dtype=bool),
        "train": times <= TRAIN_END,
        "val": (times > TRAIN_END) & (times <= VAL_END),
        "test": times > VAL_END,
    }
    tax_rows = []
    for split, sel in splits.items():
        stats = classify_natural_missing(obs[sel])
        stats["split"] = split
        tax_rows.append(stats)
        print(
            f"[taxonomy {split}] rate={stats['missing_rate']:.3f} "
            f"station={stats['frac_station']:.3f} sensor={stats['frac_sensor_ge1day']:.3f} "
            f"block={stats['frac_block_2to5steps']:.3f} point={stats['frac_point_isolated']:.3f} "
            f"non_point={stats['frac_non_point']:.3f}"
        )
    tax = pd.DataFrame(tax_rows)
    tax_path = TABLES / "natural_missing_taxonomy.csv"
    tax.to_csv(tax_path, index=False)

    Xtr, Mtr, _ = make_windows(panel, split="train")
    _, _, starts = make_windows(panel, split="test")
    mean, std = standardize_fit(Xtr, Mtr)
    panel_s = apply_standardize(panel.values, mean, std)
    panel_s[obs < 0.5] = np.nan
    y_full = np.nan_to_num(panel_s, nan=0.0)
    do_index = panel.variables.index("DO")
    test_sel = (times > VAL_END)[:, None, None]

    device = get_device()
    ckpt_path = CKPT / "maskview_spatial_plus_l0.pt"
    learned = None
    if ckpt_path.exists():
        try:
            model, ckpt = load_model(ckpt_path, device)
            adj = torch.from_numpy(ckpt["adj"]).float().to(device)
            learned = (model, adj)
            print("loaded spatial_plus_l0")
        except Exception as exc:
            print("skip spatial_plus_l0:", exc)

    rows: list[dict] = []
    for src_year in (2022, 2023):
        art = year_shift_art(times, obs, src_year, 2024)
        n_hide = int(((obs > 0.5) & (art < 0.5) & test_sel).sum())
        print(f"[replay {src_year}->2024] n_eval={n_hide}")
        if n_hide < 100:
            print("  too few overlapping holes, skip")
            continue
        x_in = panel_s.copy()
        hide = (obs > 0.5) & (art < 0.5)
        x_in[hide] = np.nan
        m_in = (~np.isnan(x_in)).astype(np.float32)
        for name, fn in (("Mean", impute_mean), ("LOCF", impute_locf), ("Linear", impute_linear)):
            pred = fn(x_in[None], m_in[None])[0]
            rec = score_art(
                name, pred, y_full, obs, art, do_index,
                "year_shift_replay", f"{src_year}_to_2024", "full_panel",
            )
            rows.append(rec)
            print(f"  {name:8s} MAE={rec['MAE']:.4f}")
        if learned is not None:
            pred = maskview_stitch(learned[0], panel_s, obs, art, starts, learned[1], device)
            rec = score_art(
                "spatial_plus_l0", pred, y_full, obs, art, do_index,
                "year_shift_replay", f"{src_year}_to_2024", "window_stitch_6day",
            )
            rows.append(rec)
            print(f"  spatial_plus_l0 MAE={rec['MAE']:.4f}")

        art_mcar = mcar_match_rate(obs, art, rng, test_sel[:, 0, 0][:, None, None])
        x_in = panel_s.copy()
        hide = (obs > 0.5) & (art_mcar < 0.5)
        x_in[hide] = np.nan
        m_in = (~np.isnan(x_in)).astype(np.float32)
        for name, fn in (("Mean", impute_mean), ("LOCF", impute_locf), ("Linear", impute_linear)):
            pred = fn(x_in[None], m_in[None])[0]
            rec = score_art(
                name, pred, y_full, obs, art_mcar, do_index,
                "mcar_matched_rate", f"{src_year}_to_2024", "full_panel",
            )
            rows.append(rec)
            print(f"  MCAR {name:8s} MAE={rec['MAE']:.4f}")
        if learned is not None:
            pred = maskview_stitch(
                learned[0], panel_s, obs, art_mcar, starts, learned[1], device
            )
            rec = score_art(
                "spatial_plus_l0", pred, y_full, obs, art_mcar, do_index,
                "mcar_matched_rate", f"{src_year}_to_2024", "window_stitch_6day",
            )
            rows.append(rec)
            print(f"  MCAR spatial_plus_l0 MAE={rec['MAE']:.4f}")

    df = pd.DataFrame(rows)
    out_csv = TABLES / "natural_missing_replay.csv"
    df.to_csv(out_csv, index=False)

    tax_md = [
        "| split | missing_rate | frac_station | frac_sensor_ge1day | frac_block | frac_point | non_point |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in tax.iterrows():
        tax_md.append(
            f"| {r['split']} | {r['missing_rate']:.3f} | {r['frac_station']:.3f} | "
            f"{r['frac_sensor_ge1day']:.3f} | {r['frac_block_2to5steps']:.3f} | "
            f"{r['frac_point_isolated']:.3f} | {r['frac_non_point']:.3f} |"
        )
    lines = [
        "# Natural missingness taxonomy and year-shift replay",
        "",
        "Naturally missing cells have no ground truth. Replay copies another year's",
        "missing calendar onto 2024 observed cells. MCAR control uses the same hide count.",
        "",
        f"Station-outage duration (all years): per-run p50={float(tax.loc[tax.split=='all','station_outage_run_p50_days'].iloc[0]):.1f} days, "
        f"p90={float(tax.loc[tax.split=='all','station_outage_run_p90_days'].iloc[0]):.1f} days; "
        f"cell-weighted p50={float(tax.loc[tax.split=='all','station_outage_cell_p50_days'].iloc[0]):.1f} days, "
        f"p90={float(tax.loc[tax.split=='all','station_outage_cell_p90_days'].iloc[0]):.1f} days "
        f"(n_runs={int(tax.loc[tax.split=='all','n_station_outage_runs'].iloc[0])}).",
        "",
        "## Taxonomy (fraction of missing cells)",
        "",
        *tax_md,
        "",
        "## Replay vs MCAR (standardized MAE)",
        "",
        "| Protocol | Setting | Linear | LOCF | spatial_plus_l0 | Winner | hide_rate |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    if not df.empty:
        for proto in ("year_shift_replay", "mcar_matched_rate"):
            for setting in sorted(df.setting.unique()):
                sub = df[(df.protocol == proto) & (df.setting == setting)]
                if sub.empty:
                    continue

                def g(m: str) -> str:
                    hit = sub[sub.model == m]
                    return f"{float(hit.MAE.iloc[0]):.3f}" if len(hit) else "-"

                scored = sub.dropna(subset=["MAE"])
                win = str(scored.loc[scored.MAE.idxmin(), "model"]) if len(scored) else ""
                hr = float(sub.hide_rate_on_obs.iloc[0])
                lines.append(
                    f"| {proto} | {setting} | {g('Linear')} | {g('LOCF')} | "
                    f"{g('spatial_plus_l0')} | {win} | {hr:.3f} |"
                )
    md = TABLES / "natural_missing_replay.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", tax_path)
    print("Wrote", out_csv)
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
