# Baseline Report

## Cross-Validation Summary

- No metrics found. Train models first.

## Recommendations

- Insufficient labels for modeling; fill metadata template and retrain.

## Notes

- All model metrics use grouped cross-validation to avoid window/video leakage.
- If window-level features were used, metrics are aggregated and reported at video level.
- Window count (e.g., 8077) is never treated as independent evaluation sample size.
- When person_id has <3 unique IDs, grouping falls back to video-level and is marked provisional.
- Severity score is derived from calibrated uncertainty and normalized feature deviations.
- Severity is not trained as an independent label target in this pipeline.