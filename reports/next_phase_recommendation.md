# Next Phase Recommendation

## What Can Be Trained Now

- No supervised task currently passes readiness thresholds.
- First realistic training run should start with `food_type` once labels are collected.
- `chew_side_label` can follow as soon as left/right labels are sufficiently populated.

## What Should Stay Rule-Based For Now

- Keep chew rate, pause ratio, rhythm variance, asymmetry ratio, jaw opening stats, and dominant frequency as direct analytics outputs.
- Do not train severity as an independent target; derive it from calibrated model uncertainty plus normalized feature deviations.
- Treat window-derived rows as segmentation artifacts, not true independent sample size.

## Label Collection Priorities

- Priority 1: `food_type` with balanced coverage across soft/medium/hard.
- Priority 2: `chew_side_label` with clear left/right labels (balanced can remain unlabeled).
- Add `person_id` consistently to enable person-level grouped validation.
- Use `label_quality` and `notes` fields to document uncertain labels.

## Window-Level Modeling Guidance

- True sample scale is `20` original videos; windows (e.g. 8077) only increase within-video observations.
- Window-level training can help feature stabilization but must keep grouped splits and video-level reporting.
- If person_id count is below 3 unique IDs, treat grouped results as provisional.

## Tasks Not Trainable Yet

- `food_type`: insufficient labeled coverage.
- `chew_side_label`: insufficient labeled coverage.
- `rhythm_class`: insufficient labeled coverage.
- `pause_style`: insufficient labeled coverage.
- `chew_rate_target`: insufficient labeled coverage.