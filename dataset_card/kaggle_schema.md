# Kaggle-style competition layout (draft)

```
dianchi-imputation/
  train/                 # 2022-01 → 2023-06 windows with natural missing
  test_input/            # 2024 windows with additional hidden cells (art mask)
  test_target/           # hidden for scoring (organizer only)
  sample_submission.csv  # flatten (window_id,t,station,var) → value
  station_distance_km.csv
  meta.json              # seq_len, vars, stations, mask patterns
```

## Scoring

Primary: MAE on artificially masked positions that were originally observed.  
Secondary: RMSE; block_time / station pattern MAE (robustness).

## Submission format

| window_id | t | station | variable | value |
|-----------|---|--------:|----------|------:|
| 0 | 0 | Baiyukou | DO | 7.21 |
