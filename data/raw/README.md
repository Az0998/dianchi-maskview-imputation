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
  - time-series
  - environmental-monitoring
  - benchmark
size_categories:
  - 100K<n<1M
---
# Dianchi Water

A high-frequency (4-hourly) multi-station surface water quality dataset
covering **22 monitoring stations**, **9 water quality variables**, and
**3 years** (2022–2024) in the Dianchi Lake basin, China.

Natural missing rate: **19.8%** (>99% block-structured).

## Contents

```
data/
  dianchi_data_df.parquet            # Main dataset (116,783 records)
  dianchi_station_distance_km.csv    # 22×22 pairwise Haversine distance (km)
scripts/
  build_adjacency.py                 # Generate adjacency matrices + heatmaps
```

## Quick start

```python
import pandas as pd

df = pd.read_parquet("data/dianchi_data_df.parquet")
print(df.shape)                 # (116783, 11)
print(df.columns.tolist())      # ['tm', 'station', 'TEM', 'PH', ...]
print(df["station"].nunique())  # 22
```

## Column descriptions

| Column   | Type             | Description                          |
|----------|------------------|--------------------------------------|
| `tm`     | datetime64\[ns\] | Timestamp (4-hourly cadence)         |
| `station`| string           | Monitoring station name (English)    |
| `TEM`    | float64          | Water temperature (°C)               |
| `PH`     | float64          | pH                                   |
| `DO`     | float64          | Dissolved oxygen (mg/L)              |
| `CON`    | float64          | Electrical conductivity (μS/cm)      |
| `NTU`    | float64          | Turbidity (NTU)                      |
| `IMN`    | float64          | Permanganate index (mg/L)            |
| `NH_N`   | float64          | Ammonia nitrogen (mg/L)              |
| `TP`     | float64          | Total phosphorus (mg/L)              |
| `TN`     | float64          | Total nitrogen (mg/L)                |

## Dataset scale

- **Records:** 116,783
- **Stations:** 22
- **Variables:** 9 target water quality variables
- **Time range:** 2022-01-01 to 2024-12-30
- **Frequency:** 4-hourly (6 observations/day)
- **Full 4h grid per station:** 6,568 time steps
- **Aggregate missing rate:** 19.8% (on full 4h grid, 9 variables)

## Station observation rates

Observation rates after reindexing to the full 4-hourly grid
(6,568 steps per station):

| Station                    | Records | Obs rate |
|----------------------------|--------:|---------:|
| Daguanhe Inlet             |   5,866 |    89.3% |
| Chuanfang Bridge           |   5,866 |    89.3% |
| Duanqiao                   |   5,848 |    89.0% |
| Caohai Center              |   5,844 |    89.0% |
| Xinhecun Inlet             |   5,837 |    88.9% |
| Guanyinshan West           |   5,788 |    88.1% |
| Wangda Bridge              |   5,767 |    87.8% |
| Huilong Village            |   5,762 |    87.7% |
| Dianchi South              |   5,758 |    87.7% |
| Luojiaying                 |   5,752 |    87.6% |
| Baofengcun Inlet           |   5,747 |    87.5% |
| Haikou West                |   5,676 |    86.4% |
| Jiangwei Lower Sluice      |   5,627 |    85.7% |
| Dayuxiang Tuluocun Inlet   |   5,512 |    83.9% |
| Huiwan Central             |   5,497 |    83.7% |
| Baiyukou                   |   5,446 |    82.9% |
| Yanjiancun Bridge          |   5,429 |    82.7% |
| Guanyinshan East           |   5,354 |    81.5% |
| Guanyinshan Central        |   4,835 |    73.6% |
| Dongdahe Dianchi Inlet     |   4,686 |    71.3% |
| Cigang River Inlet         |   2,876 |    43.8% |
| Xiyuan Tunnel              |   2,010 |    30.6% |

## Variable summary statistics

| Variable | Unit   | Missing% | Mean   | Std    | Min  | P5     | Median | P95    | Max      | Skew  |
|----------|--------|----------|--------|--------|------|--------|--------|--------|----------|-------|
| TEM      | °C     | 19.3%    | 18.73  | 4.23   | 0.00 | 11.64  | 19.12  | 24.80  | 36.10    | −0.2  |
| PH       | —      | 20.0%    | 8.26   | 0.71   | 0.00 | 7.36   | 8.30   | 9.13   | 10.99    | −4.7  |
| DO       | mg/L   | 19.3%    | 7.60   | 3.16   | 0.00 | 2.89   | 7.43   | 13.22  | 29.99    | 1.1   |
| CON      | μS/cm  | 19.3%    | 514.59 | 131.85 | 0.00 | 348.10 | 490.30 | 753.12 | 1780.86  | 1.8   |
| NTU      | NTU    | 19.5%    | 20.71  | 44.97  | 0.00 | 2.60   | 14.10  | 51.99  | 9918.65  | 103.0 |
| IMN      | mg/L   | 20.5%    | 4.64   | 2.27   | 0.00 | 1.43   | 4.36   | 8.15   | 31.51    | 0.6   |
| NH_N     | mg/L   | 20.2%    | 0.21   | 0.49   | 0.00 | 0.03   | 0.04   | 0.78   | 16.73    | 9.6   |
| TP       | mg/L   | 20.1%    | 0.08   | 0.07   | 0.00 | 0.02   | 0.07   | 0.17   | 3.55     | 10.2  |
| TN       | mg/L   | 20.1%    | 3.24   | 2.24   | 0.00 | 0.90   | 2.41   | 7.66   | 45.03    | 1.3   |

## Adjacency construction

The script `scripts/build_adjacency.py` reads the distance matrix and
constructs adjacency matrices using a linear-decay weight:

$$w_{ij} = \max\!\bigl(0,\; 1 - d_{ij} / \tau\bigr)$$

where $d_{ij}$ is the geodesic distance between stations $i$ and $j$,
and $\tau$ is a user-specified threshold in kilometres.

```bash
# Default thresholds (10, 15, 20, 25, 30 km)
python scripts/build_adjacency.py

# Single threshold
python scripts/build_adjacency.py --threshold-km 20

# Custom thresholds, custom output directory
python scripts/build_adjacency.py --thresholds-km 5,10,20 --output-dir ./outputs
```

**Dependencies:** `numpy`, `pandas`, `matplotlib`

## Privacy note

Raw station coordinates are **not** included. The pairwise distance
matrix preserves all information needed for distance-based graph
construction without exposing exact locations.

## Citation

If you use this data, please cite the accompanying paper:

```
@article{anonymous2026dianchiwater,
  title   = {A High-Frequency Multi-Station Surface Water Quality
             Dataset and Mask-View Augmentation Benchmark for
             Time-Series Imputation},
  author  = {Anonymous},
  year    = {2026},
}
```

## License

This dataset is released under the
[Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)
license.