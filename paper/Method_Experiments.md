# Method and Experiments
**Working title:** Mask-View Spatiotemporal Imputation for Block-Missing Water-Quality Monitoring Networks: A Dianchi Lake Basin Benchmark

**Target venues:** *Water* (MDPI); backup *Environmental Modelling & Software*  
**Status:** full grid complete (7 models × 5 patterns × 3 rates = 105 rows)

**Companion files**
- Word: `paper/Method_Experiments.docx`
- Full MAE grid: `results/tables/table_mae_full.csv`
- Pattern-average: `results/tables/table_mae_by_pattern_avg.csv`
- DO grid: `results/tables/table_mae_do_full.csv`
- Markdown tables: `results/tables/PAPER_TABLES.md`

---

## 2. Materials and Methods

### 2.1. Study Area and Dataset

We use the open **Dianchi-Water** high-frequency surface-water quality dataset covering the Dianchi Lake basin (Kunming, Yunnan, China). The release provides 4-hourly observations from **22 automatic monitoring stations** over **2022-01-01 to 2024-12-30**, with nine physicochemical variables:

TEM, pH, DO, CON, NTU, IMN, NH₃-N, TP, TN.

After alignment to a regular 4 h grid, the aggregate missing rate is approximately **19.8%**, and missingness is predominantly **block-structured** (sensor downtime and station outages). A pairwise Haversine **station distance matrix** accompanies the release and supports spatial graph construction.

### 2.2. Spatiotemporal Panel, Windows and Split

Let \(N=22\) stations and \(D=9\) variables. We form \(X\in\mathbb{R}^{T\times N\times D}\) with observation mask \(M\). Sliding windows of length \(L=36\) (6 days) and stride \(S=18\) yield samples of shape \((L,N,D)\).

**Temporal split (no leakage):** train ≤ 2023-06-30; validation 2023-07–2023-12; test 2024.  
Variables are standardized using training-set means/stds on observed cells. Multivariate metrics are reported in **standardized space**; DO-specific scores are reported separately.

### 2.3. Spatial Graph

From distances \(D_{ij}\) (km) we build a \(k\)-NN Gaussian adjacency (\(k=3\), cutoff \(40\,\mathrm{km}\), \(\sigma=15\,\mathrm{km}\)), symmetrize and row-normalize, retaining self-loops.

### 2.4. Mask-View Augmentation (Proposed)

Automatic-station failures are poorly described by MCAR holes alone. **Mask-View** trains the imputer on \(K=3\) complementary artificially masked views of the same window.

**Pattern bank**

| Pattern | Meaning | Construction |
|---------|---------|--------------|
| point | sporadic glitches | MCAR on observed cells |
| block_time | downtime | consecutive temporal slabs |
| sensor | parameter failure | drop 1–2 variables |
| station | station outage | drop 1–3 stations |
| mixed | compound | point + block (+ optional station) |

Training samples artificial ratio \(r\sim U[0.15,0.35]\). Evaluation uses \(r\in\{0.1,0.2,0.3\}\) under all five patterns.

**Losses.** Reconstruction \(\mathcal{L}_{\mathrm{rec}}\) on originally observed cells that were artificially hidden; cross-view consistency \(\mathcal{L}_{\mathrm{cons}}\) on cells visible in multiple views; total \(\mathcal{L}=\mathcal{L}_{\mathrm{rec}}+\lambda_{\mathrm{cons}}\mathcal{L}_{\mathrm{cons}}\) with \(\lambda_{\mathrm{cons}}=0.5\).

### 2.5. MaskView-ST Architecture

**MaskView-ST** stacks two spatiotemporal blocks: graph convolution over stations + multi-head temporal self-attention along time. Inputs concatenate values and the current mask (\(2D\) channels). An MLP decoder predicts all channels; observed entries are kept at inference. Capacity: hidden 64, 4 heads, AdamW \(10^{-3}\), batch 32, **40 epochs** with validation checkpointing.

### 2.6. Baselines

Mean; LOCF; Linear (per station–variable); SAITS; BRITS; StemGNN reported as **GRIN\*** (official GRIN unavailable in PyPOTS 1.5). Each deep baseline is fit **once** on training windows, then evaluated on every test corruption.

### 2.7. Evaluation Protocol

Artificial hides apply only to originally observed cells. We report MAE/RMSE/MAPE (standardized) and MAE/RMSE for DO. Primary scientific claims use average MAE, hard-pattern MAE (block/sensor/station/mixed), and DO MAE.

---

## 3. Results

### 3.1. Headline Rankings (105 settings)

| Ranking criterion | Best | Score |
|-------------------|------|------:|
| Mean MAE (all settings) | **MaskView-ST** | **0.388** |
| Mean DO MAE | **MaskView-ST** | **0.480** |
| Hard-pattern mean MAE | **MaskView-ST** | **0.411** |
| Per-setting MAE wins | Linear (9), BRITS (5), MaskView-ST (1) | — |

MaskView-ST is the best method on **average error**, **DO**, and **structured-missing** averages, even though Linear wins the largest number of individual easy settings (point / short-gap regimes).

### 3.2. Pattern-Averaged Standardized MAE

Averaged over rates \(10\%/20\%/30\%\):

| Model | point | block_time | sensor | station | mixed |
|-------|------:|-----------:|-------:|--------:|------:|
| Mean | 0.687 | 0.686 | 0.686 | 0.673 | 0.683 |
| LOCF | 0.206 | 0.291 | 0.696 | 0.679 | 0.299 |
| Linear | **0.161** | **0.254** | 0.696 | 0.679 | **0.264** |
| SAITS | 0.446 | 0.451 | 0.449 | 0.444 | 0.445 |
| BRITS | 0.429 | 0.442 | 0.430 | **0.435** | 0.433 |
| GRIN* | 0.459 | 0.463 | 0.459 | 0.457 | 0.457 |
| **MaskView-ST** | 0.297 | 0.326 | **0.441** | 0.539 | 0.339 |

**Interpretation**
1. **Point / mixed / block_time:** Linear remains strongest—4 h series are locally smooth, so temporal interpolation is a high bar.  
2. **Sensor dropout:** Linear/LOCF collapse to ~0.70 (no within-series support); MaskView-ST (0.441) and BRITS (0.430) recover cross-variable/spatial signal.  
3. **Station outage:** classical methods again fail (~0.68); deep models help. BRITS/SAITS currently edge MaskView-ST on multivariate station MAE, while MaskView-ST remains best overall and on DO averages.  
4. **Among deep models,** MaskView-ST dominates point/block/mixed and is competitive on sensor.

### 3.3. Dissolved Oxygen (standardized MAE)

MaskView-ST achieves the best **mean DO MAE (0.480)** versus BRITS (0.522), SAITS (0.561) and Linear (0.573). Representative cells:

| Setting | Linear | BRITS | MaskView-ST |
|---------|-------:|------:|------------:|
| point@20% | 0.385 | 0.507 | 0.396 |
| block_time@20% | 0.579 | 0.537 | **0.466** |
| sensor@20% | 0.712 | 0.537 | **0.494** |
| station@20% | 0.718 | **0.547** | 0.644 |
| mixed@20% | 0.503 | 0.527 | **0.457** |

For hypoxia-oriented use, MaskView-ST is the preferred deep imputer under block/sensor/mixed corruption.

### 3.4. Full Grid

The complete 15-setting × 7-model MAE/RMSE/DO tables are archived at:

- `results/tables/table_mae_full.csv`
- `results/tables/table_rmse_full.csv`
- `results/tables/table_mae_do_full.csv`
- `results/tables/PAPER_TABLES.md`

Figures: `fig_imputation_mae_heatmap.png`, `fig_maskview_patterns.png`, `fig_imputation_example_DO.png`.

### 3.5. What the Results Mean for the Paper Claim

The correct claim is **not** “MaskView beats Linear everywhere.” It is:

> Under a protocol that includes operational structured missingness, Mask-View spatiotemporal imputation achieves the best **average** and **DO** accuracy, and substantially outperforms temporal interpolators when sensors or stations fail—precisely the failure mode of automatic monitoring networks.

This matches the Dianchi missingness structure (>99% block-structured in the dataset card) and justifies Mask-View as the methodological contribution.

---

## 3.6. Ablations and Spatial Upgrade

We fixed a consistency-loss implementation issue (consistency must be computed on **raw decoder outputs**, not on blended imputations that equal the input on visible cells). After the fix:

### λ_cons

| Variant | Overall MAE | Station MAE |
|---------|------------:|------------:|
| λ=0 (`lcons0_fixed`) | **0.397** | 0.543 |
| λ=0.5 (`lcons05_fixed`) | 0.416 | **0.539** |
| λ=1.0 (`lcons10_fixed`) | 0.447 | 0.545 |

Higher consistency weight **hurts** average accuracy at the current data/model scale; mild or zero λ is preferred. Station MAE is nearly flat across λ.

### Pattern bank vs point-only

| Variant | point | sensor | station |
|---------|------:|-------:|--------:|
| `point_only` | **0.241** | 0.580 | 0.639 |
| full bank (λ=0) | 0.310 | **0.449** | **0.543** |

Training only on MCAR point masks overfits short gaps and **collapses under sensor/station outages**—direct evidence that Mask-View’s pattern bank is necessary for operational missingness.

### Spatial upgrade (`spatial_plus_l0`)

MixHop (1-/2-hop) GNN, mask-aware neighbor aggregation, denser kNN (\(k=8\)), station-pattern upsampling, **λ=0**:

| Metric | BRITS | spatial_plus_l0 | Δ |
|--------|------:|----------------:|--:|
| Overall MAE | 0.434 | **0.381** | −0.053 |
| Overall DO MAE | 0.522 | **0.459** | −0.063 |
| Sensor MAE | 0.430 | **0.421** | −0.009 |
| Station MAE | **0.435** | 0.540 | +0.105 |

**Takeaway:** spatial_plus_l0 is the best overall / DO / sensor recipe and narrows—but does not eliminate—the station gap to BRITS. Remaining gap is consistent with BRITS operating on flattened multi-station feature channels with stronger cross-node mixing than a local graph. Tables/figures: `results/tables/ablation_*.csv`, `fig_ablation_*.png`.

## 4. Limitations and Next Revisions

1. **GRIN\*** is StemGNN, not official GRIN—replace before camera-ready if possible.  
2. **Station-pattern gap** vs BRITS remains (~0.10 MAE); try multi-scale graphs or station-conditioned decoders.  
3. Downstream warning (DO drop / TN exceedance): `run_downstream.py`.

---

## Reproducibility

```bash
python run_benchmark.py --epochs 40 --pypots-epochs 20
python export_paper_tables.py
python build_method_experiments_docx.py
```

Seed = 42. Dataset: `anonymous-dianchi-2026/dianchi-water`.
