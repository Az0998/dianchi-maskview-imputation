# Natural missingness taxonomy and year-shift replay

Naturally missing cells have no ground truth. Replay copies another year's
missing calendar onto 2024 observed cells. MCAR control uses the same hide count.

Station-outage duration (all years): per-run p50=0.2 days, p90=0.8 days; cell-weighted p50=3.5 days, p90=630.5 days (n_runs=6263).

## Taxonomy (fraction of missing cells)

| split | missing_rate | frac_station | frac_sensor_ge1day | frac_block | frac_point | non_point |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.198 | 0.969 | 0.019 | 0.006 | 0.006 | 0.994 |
| train | 0.257 | 0.962 | 0.027 | 0.006 | 0.006 | 0.994 |
| val | 0.130 | 0.978 | 0.003 | 0.010 | 0.010 | 0.990 |
| test | 0.143 | 0.985 | 0.007 | 0.004 | 0.004 | 0.996 |

## Replay vs MCAR (standardized MAE)

| Protocol | Setting | Linear | LOCF | spatial_plus_l0 | Winner | hide_rate |
|---|---|---:|---:|---:|---|---:|
| year_shift_replay | 2022_to_2024 | 0.383 | 0.495 | 0.494 | Linear | 0.105 |
| year_shift_replay | 2023_to_2024 | 0.328 | 0.403 | 0.474 | Linear | 0.035 |
| mcar_matched_rate | 2022_to_2024 | 0.168 | 0.217 | 0.297 | Linear | 0.105 |
| mcar_matched_rate | 2023_to_2024 | 0.147 | 0.198 | 0.291 | Linear | 0.035 |
