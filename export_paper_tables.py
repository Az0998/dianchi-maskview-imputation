"""Export paper-ready pivot tables and summary stats from benchmark CSV."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"
RESULTS = ROOT / "results"


def main():
    src = TABLES / "imputation_benchmark.csv"
    df = pd.read_csv(src)
    df = df[df["pattern"] != "error"].copy()
    TABLES.mkdir(parents=True, exist_ok=True)

    # Main MAE pivot: model × (pattern@rate)
    df["setting"] = df["pattern"] + "@" + df["rate"].map(lambda x: f"{x:.0%}")
    mae = df.pivot_table(index="model", columns="setting", values="MAE", aggfunc="mean")
    rmse = df.pivot_table(index="model", columns="setting", values="RMSE", aggfunc="mean")
    mae_do = df.pivot_table(index="model", columns="setting", values="MAE_DO", aggfunc="mean")

    # Order columns nicely
    patterns = ["point", "block_time", "sensor", "station", "mixed"]
    rates = [0.1, 0.2, 0.3]
    col_order = [f"{p}@{r:.0%}" for p in patterns for r in rates if f"{p}@{r:.0%}" in mae.columns]
    mae = mae.reindex(columns=col_order)
    rmse = rmse.reindex(columns=col_order)
    mae_do = mae_do.reindex(columns=col_order)

    model_order = ["Mean", "LOCF", "Linear", "SAITS", "BRITS", "GRIN*", "CSDI", "MaskView-ST"]
    model_order = [m for m in model_order if m in mae.index] + [
        m for m in mae.index if m not in model_order
    ]
    mae = mae.reindex(model_order)
    rmse = rmse.reindex(model_order)
    mae_do = mae_do.reindex(model_order)

    mae.to_csv(TABLES / "table_mae_full.csv", float_format="%.4f")
    rmse.to_csv(TABLES / "table_rmse_full.csv", float_format="%.4f")
    mae_do.to_csv(TABLES / "table_mae_do_full.csv", float_format="%.4f")

    # Compact: average over rates per pattern
    avg_pat = df.groupby(["model", "pattern"])[["MAE", "RMSE", "MAE_DO"]].mean().reset_index()
    avg_pat_mae = avg_pat.pivot(index="model", columns="pattern", values="MAE").reindex(model_order)
    avg_pat_mae = avg_pat_mae.reindex(columns=[c for c in patterns if c in avg_pat_mae.columns])
    avg_pat_mae.to_csv(TABLES / "table_mae_by_pattern_avg.csv", float_format="%.4f")

    # Best model counts / wins
    wins = []
    for setting, g in df.groupby("setting"):
        best = g.loc[g["MAE"].idxmin()]
        wins.append({"setting": setting, "best_model": best["model"], "MAE": best["MAE"]})
    wins_df = pd.DataFrame(wins)
    wins_df.to_csv(TABLES / "table_best_model_per_setting.csv", index=False)

    # Deep-only vs Linear gap on block/station
    hard = df[df["pattern"].isin(["block_time", "station", "sensor", "mixed"])]
    summary = {
        "n_rows": int(len(df)),
        "models": sorted(df["model"].unique().tolist()),
        "patterns": sorted(df["pattern"].unique().tolist()),
        "rates": sorted(df["rate"].unique().tolist()),
        "mean_mae_by_model": df.groupby("model")["MAE"].mean().sort_values().to_dict(),
        "mean_mae_do_by_model": df.groupby("model")["MAE_DO"].mean().sort_values().to_dict(),
        "hard_patterns_mean_mae": hard.groupby("model")["MAE"].mean().sort_values().to_dict(),
        "wins": wins_df["best_model"].value_counts().to_dict(),
    }
    with open(RESULTS / "paper_tables_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Markdown preview (no tabulate dependency)
    def df_to_md(frame: pd.DataFrame) -> str:
        cols = [""] + [str(c) for c in frame.columns]
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for idx, row in frame.iterrows():
            vals = [str(idx)] + [
                f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.tolist()
            ]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    md = [
        "# Paper tables (auto-exported)\n",
        "## MAE (standardized) — full\n",
        df_to_md(mae),
        "\n\n## MAE averaged over rates\n",
        df_to_md(avg_pat_mae),
        "\n\n## DO MAE (standardized) — full\n",
        df_to_md(mae_do),
        "\n",
    ]
    (TABLES / "PAPER_TABLES.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote pivots to", TABLES)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
