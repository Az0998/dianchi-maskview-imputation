# Results & Journal Positioning Summary

## Recommended journal

**Primary: *Water* (MDPI)** — water+ML fit, soundness-based review, relatively high acceptance, fast turnaround, open data/code friendly. APC applies.

**Backups:** *Environmental Research: Water*; *Journal of Hydroinformatics*.

## Headline results (standardized MAE)

| Model | Overall | DO | Hard patterns |
|-------|--------:|---:|--------------:|
| MaskView-ST | **0.388** | **0.480** | **0.411** |
| Linear | 0.411 | 0.573 | 0.473 |
| BRITS | 0.434 | 0.522 | 0.435 |

**Recommended recipe `spatial_plus_l0`:** MixHop + k=8 + mask-aware + station boost + λ=0  
→ Overall **0.381**, DO **0.459**, sensor **0.421** (beats BRITS 0.430). Station still ~0.540 vs BRITS 0.435.

## Narrative for the paper

- Claim **conditional information value** under operational block/sensor/station missingness — not “beats Linear everywhere.”
- Linear wins many **point/short-gap** settings; deep/Mask-View recovers on **sensor/station**.
- Pattern bank is necessary (`point_only` collapses on sensor/station).
- After fixing consistency on raw decoder preds, **λ_cons = 0** is best; λ↑ hurts average MAE.

## Deliverables

| Item | Path |
|------|------|
| Final Water manuscript | `paper/Water_MaskView_Dianchi_manuscript.docx` |
| Cover letter / highlights | `paper/cover_letter.txt`, `paper/highlights.txt` |
| Submission checklist | `paper/SUBMISSION_CHECKLIST_WATER.md` |
| Project site | `docs/index.html` → GitHub Pages |
| Benchmark tables | `results/tables/` |
| Figures | `results/figures/` |

## Reproduce

```bash
pip install -r requirements.txt
set HF_ENDPOINT=https://hf-mirror.com   # if in CN
python run_benchmark.py --epochs 40 --pypots-epochs 20
python export_paper_tables.py
python build_water_manuscript.py
```
