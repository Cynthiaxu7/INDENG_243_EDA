# Label Progress Report

- Total samples: **20**

## Food Type Priority

- soft/medium/hard counts: `{'soft': 0, 'medium': 0, 'hard': 0}`
- deficits to unlock target (4/class): `{'soft': 4, 'medium': 4, 'hard': 4}`
- deficits to recommended target (5/class): `{'soft': 5, 'medium': 5, 'hard': 5}`
- minimum additional labels needed to unlock: **12**
- recommendation: **Goal: reach 4 soft, 4 medium, 4 hard first. Preferred: reach 5 per class for a more stable first run.**

## Task Status Table

| task | labeled count | class balance / readiness | blocker | next action |
|---|---:|---|---|---|
| food_type | 0 | {'soft': 0, 'medium': 0, 'hard': 0} (blocked) | class coverage | label classes: soft+4, medium+4, hard+4 |
| chew_side_label | 0 | {'left': 0, 'right': 0} (blocked) | class coverage | label classes: left+5, right+5 |
| rhythm_class | 0 | {'steady': 0, 'irregular': 0} (blocked) | class coverage | label classes: steady+5, irregular+5 |
| pause_style | 0 | {'continuous': 0, 'pause-heavy': 0} (blocked) | class coverage | label classes: continuous+5, pause-heavy+5 |
| chew_rate_target | 0 | var=nan (blocked) | insufficient numeric labels | add at least 15 valid targets with variance |

## Person ID Completeness

- rows with missing person_id: **0** / 20
- unique non-empty person_id values: **20**
- can use person-level grouping now: **True**
- note: Person-level grouping is available, but repeated-person information is limited; treat results as preliminary cross-subject pilot findings.

## Label Quality Checks

- label_quality distribution: `{'<missing>': 20}`
- rows with low quality labels: **0**
- rows with missing label_quality: **20**
- review-needed rows: **20**
- readiness note: Label quality coverage is sparse. Fill label_quality during the same sprint: high = confident label, medium = mostly confident, low = ambiguous / poor-quality clip.

## Next Best Rows To Label

Top deterministic suggestions (no label guessing):

| sample_id | video_id | person_id | food_type | chew_side_label | label_quality | priority_score | reasons |
|---|---|---|---|---|---|---:|---|
| 1 | 1 | 1 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |
| 10 | 10 | 10 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |
| 11 | 11 | 11 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |
| 12 | 12 | 12 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |
| 13 | 13 | 13 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |
| 14 | 14 | 14 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |
| 15 | 15 | 15 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |
| 16 | 16 | 16 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |
| 17 | 17 | 17 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |
| 18 | 18 | 18 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |
| 19 | 19 | 19 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |
| 2 | 2 | 2 |  |  |  | 8.50 | missing chew_side_label; missing food_type; missing label_quality |