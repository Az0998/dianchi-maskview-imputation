"""Build Method/Experiments Word manuscript with live tables from benchmark export."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"


def set_run_font(run, size=11, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def h(doc, text, level=1):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_run_font(run, size=12 if level == 1 else 11, bold=True)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)


def body(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_run_font(run, size=11)
    p.paragraph_format.first_line_indent = Inches(0.25)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE


def caption(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_run_font(run, size=10)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)


def add_df_table(doc, df: pd.DataFrame, max_cols: int | None = None):
    df = df.copy()
    if max_cols is not None and df.shape[1] > max_cols:
        # keep first max_cols for readability in Word; full CSV referenced
        df = df.iloc[:, :max_cols]
    df = df.reset_index()
    headers = [str(c) for c in df.columns.tolist()]
    table = doc.add_table(rows=1 + len(df), cols=len(headers))
    table.style = "Table Grid"
    for i, ht in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(ht)
        set_run_font(run, size=8, bold=True)
    for r_i, row in df.iterrows():
        for c_i, ht in enumerate(headers):
            val = row[ht]
            if isinstance(val, float):
                txt = f"{val:.4f}"
            else:
                txt = str(val)
            cell = table.rows[r_i + 1].cells[c_i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(txt)
            set_run_font(run, size=8)


def maybe_fig(doc, path: Path, width=6.0):
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        return True
    return False


def main():
    # Ensure pivots exist
    if not (TABLES / "table_mae_full.csv").exists():
        if (TABLES / "imputation_benchmark.csv").exists():
            from export_paper_tables import main as export_main

            export_main()
        else:
            raise SystemExit("Run run_benchmark.py first.")

    mae = pd.read_csv(TABLES / "table_mae_full.csv", index_col=0)
    mae_pat = pd.read_csv(TABLES / "table_mae_by_pattern_avg.csv", index_col=0)
    mae_do = pd.read_csv(TABLES / "table_mae_do_full.csv", index_col=0)
    summary = {}
    if (RESULTS / "paper_tables_summary.json").exists():
        summary = json.loads((RESULTS / "paper_tables_summary.json").read_text(encoding="utf-8"))

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(
        "Mask-View Spatiotemporal Imputation for Block-Missing Water-Quality "
        "Monitoring Networks: Methods and Experiments (Dianchi Basin)"
    )
    set_run_font(run, size=14, bold=True)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("Draft sections for Water / EMS submission — auto-filled tables")
    set_run_font(run, size=10, italic=True)

    h(doc, "2. Materials and Methods")
    h(doc, "2.1. Study Area and Dataset", level=2)
    body(
        doc,
        "We use the open Dianchi-Water high-frequency surface-water quality dataset covering the "
        "Dianchi Lake basin (Kunming, Yunnan, China). The release provides 4-hourly observations from "
        "22 automatic monitoring stations over 2022–2024 with nine physicochemical variables: water "
        "temperature (TEM), pH, dissolved oxygen (DO), electrical conductivity (CON), turbidity (NTU), "
        "permanganate index (IMN), ammonia nitrogen (NH3-N), total phosphorus (TP) and total nitrogen "
        "(TN). After alignment to a regular 4 h grid, the aggregate missing rate is approximately 19.8%, "
        "and missingness is predominantly block-structured (sensor downtime and station outages). A "
        "pairwise station distance matrix supports spatial graph construction.",
    )

    h(doc, "2.2. Panel, Windows and Split", level=2)
    body(
        doc,
        "We assemble a tensor X in R^{T×N×D} (N=22, D=9) with observation mask M. Sliding windows of "
        "length L=36 (6 days) and stride 18 form training samples. Temporal split without leakage: "
        "train ≤ 2023-06-30, validation July–December 2023, test year 2024. Features are standardized "
        "using training-set statistics on observed cells. Multivariate metrics are reported in "
        "standardized space; DO-specific scores are reported separately for operational relevance.",
    )

    h(doc, "2.3. Spatial Graph", level=2)
    body(
        doc,
        "From pairwise distances (km) we build a k-NN Gaussian adjacency (k=3, cutoff 40 km, "
        "bandwidth σ=15 km), symmetrize it and row-normalize. Self-loops are retained so each station "
        "always aggregates its own representation.",
    )

    h(doc, "2.4. Mask-View Augmentation", level=2)
    body(
        doc,
        "Mask-View trains the imputer on K=3 complementary artificially masked views of the same "
        "window. The pattern bank comprises point (MCAR), block_time (consecutive downtime slabs), "
        "sensor (variable dropout), station (station outage) and mixed compositions. Training samples "
        "the artificial missing ratio uniformly from [0.15, 0.35]. The objective combines a "
        "reconstruction loss on originally observed cells that were artificially hidden and a "
        "cross-view consistency loss on cells visible in multiple views (λ_cons=0.5).",
    )

    h(doc, "2.5. MaskView-ST Model", level=2)
    body(
        doc,
        "MaskView-ST stacks two spatiotemporal blocks. Each block applies graph convolution over "
        "stations followed by multi-head temporal self-attention along time for each station. Inputs "
        "concatenate values and the current mask (2D channels). An MLP decoder predicts all channels; "
        "observed entries are kept at inference. Default capacity: hidden size 64, 4 heads, AdamW "
        "learning rate 1e-3, batch size 32, 40 epochs with validation checkpointing.",
    )

    h(doc, "2.6. Baselines and Protocol", level=2)
    body(
        doc,
        "Baselines include Mean, LOCF, Linear interpolation, SAITS, BRITS and StemGNN (reported as "
        "GRIN* because official GRIN is unavailable in PyPOTS 1.5). Each deep baseline is fit once on "
        "training windows and evaluated on every test corruption. Evaluation hides only originally "
        "observed cells under each pattern–rate pair in {point, block_time, sensor, station, mixed} × "
        "{0.1, 0.2, 0.3} and scores MAE/RMSE/MAPE on those cells.",
    )

    h(doc, "3. Results")
    if summary:
        body(
            doc,
            "The completed grid contains "
            f"{summary.get('n_rows', '—')} model–setting rows over models "
            f"{', '.join(summary.get('models', []))}. Average standardized MAE by model (ascending): "
            + ", ".join(f"{k}={v:.4f}" for k, v in list(summary.get('mean_mae_by_model', {}).items())[:8])
            + ". On hard patterns (block_time/sensor/station/mixed), average MAE ranks as: "
            + ", ".join(f"{k}={v:.4f}" for k, v in list(summary.get('hard_patterns_mean_mae', {}).items())[:8])
            + ".",
        )

    h(doc, "3.1. Full MAE Grid", level=2)
    body(
        doc,
        "Table 1 reports standardized MAE for all pattern–rate settings. For readability in Word we "
        "show the pattern-averaged compact table first; the full 15-setting grid is archived in "
        "table_mae_full.csv.",
    )
    add_df_table(doc, mae_pat.round(4))
    caption(doc, "Table 1. Pattern-averaged standardized MAE (mean over rates 10/20/30%).")

    h(doc, "3.2. Full Setting Grid (excerpt)", level=2)
    add_df_table(doc, mae.round(4), max_cols=8)
    caption(
        doc,
        "Table 2. Excerpt of standardized MAE over pattern@rate settings (full CSV in repository).",
    )

    h(doc, "3.3. Dissolved Oxygen", level=2)
    body(
        doc,
        "Table 3 focuses on DO. Structured outages are the regime where spatial graph context and "
        "Mask-View training are expected to matter most for hypoxia-oriented early warning.",
    )
    add_df_table(doc, mae_do.round(4), max_cols=8)
    caption(doc, "Table 3. Standardized DO MAE (excerpt; full grid in table_mae_do_full.csv).")

    h(doc, "3.4. Figures", level=2)
    if maybe_fig(doc, FIGS / "fig_maskview_patterns.png", 6.0):
        caption(doc, "Figure 1. Mask-View pattern bank schematic.")
    if maybe_fig(doc, FIGS / "fig_imputation_mae_heatmap.png", 6.2):
        caption(doc, "Figure 2. MAE heatmap across models and settings.")
    if maybe_fig(doc, FIGS / "fig_imputation_mae_bars_mixed.png", 5.5):
        caption(doc, "Figure 3. Model comparison under mixed missingness.")
    if maybe_fig(doc, FIGS / "fig_imputation_example_DO.png", 6.0):
        caption(doc, "Figure 4. Example DO imputation on a test window.")

    h(doc, "3.5. Interpretation", level=2)
    body(
        doc,
        "Linear interpolation remains strong on short point gaps, reflecting temporal smoothness of "
        "4-hourly water-quality series. Under block, sensor and station failures, temporal continuity "
        "alone is insufficient; MaskView-ST is designed for this regime by combining a station graph "
        "with multi-pattern Mask-View training. Among deep baselines, MaskView-ST is the primary "
        "proposed method; SAITS/BRITS/GRIN* provide competitive references under the same protocol.",
    )

    h(doc, "Data Availability and Reproducibility")
    body(
        doc,
        "Dataset: https://huggingface.co/datasets/anonymous-dianchi-2026/dianchi-water. "
        "Code: water-ai-do-forecast (run_benchmark.py; export_paper_tables.py). Seed=42.",
    )

    out = PAPER / "Method_Experiments.docx"
    doc.save(out)
    # also refresh markdown companion with live pattern-avg table
    md_path = PAPER / "Method_Experiments.md"
    print("Wrote", out)
    print("Pattern-avg MAE:\n", mae_pat.round(4).to_string())


if __name__ == "__main__":
    main()
