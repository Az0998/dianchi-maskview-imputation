---
license: cc-by-4.0
task_categories:
  - time-series-forecasting
  - tabular-regression
language:
  - en
tags:
  - water-quality
  - imputation
  - mask-view
  - spatiotemporal
  - environmental-monitoring
  - dianchi
  - benchmark
size_categories:
  - 100K<n<1M
pretty_name: Dianchi-Water Mask-View Benchmark
---

# Dianchi-Water for Spatiotemporal Imputation

High-frequency (4-hourly) multi-station surface water quality data for the **Dianchi Lake basin**, China.

| Item | Value |
|------|------:|
| Stations | 22 |
| Variables | 9 |
| Records | 116,783 |
| Period | 2022-01-01 → 2024-12-30 |
| Natural missing rate (full grid) | ~19.8% |
| Missing structure | >99% block-structured |

## Why this dataset

Most imputation benchmarks assume MCAR holes. Dianchi automatic stations fail as **blocks** (sensor downtime / station outage). This release includes a **station distance matrix** so models can exploit spatial neighbors — exactly when a station goes dark.

## Files

```
data/dianchi_data_df.parquet
data/dianchi_station_distance_km.csv
```

## Variables

`TEM, PH, DO, CON, NTU, IMN, NH_N, TP, TN`

## Recommended tasks

1. **Spatiotemporal imputation** under point / block_time / sensor / station masks  
2. **Mask-View** robustness benchmarking  
3. Downstream **DO drop / TN exceedance** early warning after imputation  

## Quick load

```python
import pandas as pd
df = pd.read_parquet("data/dianchi_data_df.parquet")
dist = pd.read_csv("data/dianchi_station_distance_km.csv", index_col=0)
```

## Citation

Please cite the dataset page and this benchmark repository when using Mask-View results.

## License

CC BY 4.0
