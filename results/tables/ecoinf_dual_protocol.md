# Dual protocol: EcoInf 2025 missingness vs Mask-View operational bank

Same Dianchi panel. EcoInf: random 20-80% and 1-4 week gaps (doi:10.1016/j.ecoinf.2025.103283).
Learned EcoInf numbers are spatial_plus_l0 (current MixHop checkpoint). maskview_st.pt is an older graph block and was skipped.
Week-gap Linear/LOCF see the full series; spatial_plus_l0 is stitched 6-day windows.

| Protocol | Setting | Linear | LOCF | MaskView-ST | spatial_plus_l0 | Winner |
|---|---|---:|---:|---:|---:|---|
| ecoinf_random | random_20pct | 0.160 | 0.205 | - | 0.292 | Linear |
| ecoinf_random | random_40pct | 0.183 | 0.228 | - | 0.306 | Linear |
| ecoinf_random | random_60pct | 0.209 | 0.247 | - | 0.340 | Linear |
| ecoinf_random | random_80pct | 0.242 | 0.275 | - | 0.408 | Linear |
| ecoinf_week_gap | 1week | 0.311 | 0.357 | - | 0.349 | Linear |
| ecoinf_week_gap | 2week | 0.285 | 0.340 | - | 0.378 | Linear |
| ecoinf_week_gap | 3week | 0.341 | 0.416 | - | 0.401 | Linear |
| ecoinf_week_gap | 4week | 0.402 | 0.475 | - | 0.427 | Linear |
| maskview_operational | point | 0.161 | 0.206 | 0.297 | 0.295 | Linear |
| maskview_operational | block_time | 0.254 | 0.290 | 0.326 | 0.313 | Linear |
| maskview_operational | sensor | 0.696 | 0.696 | 0.441 | 0.421 | spatial_plus_l0 |
| maskview_operational | station | 0.679 | 0.679 | 0.539 | 0.540 | BRITS |

Linear wins all 8 EcoInf settings. Operational sensor/station flip the ranking.

