# Direct Analytics Report

- This report uses threshold-based analytics outputs, not supervised targets.
- Severity should be derived later from uncertainty + normalized deviations.

## Dataset Summary

- Number of videos: **20**
- Median chew_rate_hz: **5.365**
- Median pause_ratio: **0.369**

## Categorical Band Counts

- `chew_rate_band`: {'normal': 10, 'fast': 5, 'slow': 5}
- `asymmetry_band`: {'strong bias': 10, 'mild bias': 9, 'balanced': 1}
- `pause_style_band`: {'moderate pauses': 11, 'pause-heavy': 5, 'continuous': 4}
- `rhythm_band`: {'irregular-like': 7, 'steady-like': 7, 'moderate variability': 6}

## Output Files

- Per-video analytics table: `artifacts/results/per_video_analytics.csv`