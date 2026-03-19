# Food Type Sprint Report

- Suggested rows are for **diverse review coverage only**.
- They are **not predicted class memberships**.

- Current counts (soft/medium/hard): `{'soft': 0, 'medium': 0, 'hard': 0}`
- Unlock target deficits (4/class): `{'soft': 4, 'medium': 4, 'hard': 4}`
- Recommended target deficits (5/class): `{'soft': 5, 'medium': 5, 'hard': 5}`
- Minimum labels to unlock: **12**
- Recommendation: **Goal: reach 4 soft, 4 medium, 4 hard first. Preferred: reach 5 per class for a more stable first run.**

## Diverse Review Queue

| sample_id | video_id | person_id | chew_rate_hz | asymmetry_ratio | pause_ratio | rhythm_variance | jaw_opening_mean | dominant_frequency | priority_score | priority_reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 8 | 8 | 5.3183 | 0.9924 | 0.5288 | 1518.2163 | 22.3770 | 0.0049 | 9.23 | coverage spread in jaw_opening_mean; coverage spread in rhythm_variance; diverse asymmetry_band; diverse pause_style_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 2 | 2 | 2 | 2.9260 | 0.6987 | 0.4941 | 2010.7773 | 26.8894 | 0.0036 | 8.91 | coverage spread in asymmetry_ratio; coverage spread in jaw_opening_mean; coverage spread in rhythm_variance; diverse chew_rate_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 12 | 12 | 12 | 5.7058 | 1.0850 | 0.5368 | 2311.7444 | 26.6896 | 0.0092 | 8.57 | coverage spread in asymmetry_ratio; coverage spread in jaw_opening_mean; coverage spread in pause_ratio; coverage spread in rhythm_variance; diverse asymmetry_band; diverse pause_style_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 7 | 7 | 7 | 3.9288 | 0.8367 | 0.2367 | 1429.3132 | 25.6621 | 0.0028 | 8.56 | coverage spread in asymmetry_ratio; coverage spread in jaw_opening_mean; coverage spread in rhythm_variance; diverse pause_style_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 6 | 6 | 6 | 3.1631 | 1.0015 | 0.2459 | 556.0066 | 6.7213 | 0.0851 | 8.49 | coverage spread in dominant_frequency; diverse asymmetry_band; diverse pause_style_band; diverse rhythm_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 3 | 3 | 3 | 2.8138 | 1.0337 | 0.1780 | 97.6289 | 5.8407 | 0.0419 | 8.41 | coverage spread in dominant_frequency; coverage spread in pause_ratio; diverse chew_rate_band; diverse pause_style_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 9 | 9 | 9 | 11.8210 | 0.9849 | 0.6562 | 303.6767 | 6.3230 | 0.0079 | 8.37 | coverage spread in chew_rate_hz; coverage spread in pause_ratio; diverse asymmetry_band; diverse chew_rate_band; diverse pause_style_band; diverse rhythm_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 15 | 15 | 15 | 11.3790 | 1.0589 | 0.5480 | 148.3806 | 6.0068 | 0.0039 | 8.30 | coverage spread in chew_rate_hz; coverage spread in pause_ratio; diverse chew_rate_band; diverse pause_style_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 5 | 5 | 5 | 2.6479 | 0.9586 | 0.3700 | 69.5658 | 3.5223 | 0.0562 | 8.19 | coverage spread in dominant_frequency; diverse chew_rate_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 11 | 11 | 11 | 9.3202 | 1.0056 | 0.4914 | 1038.9823 | 21.7212 | 0.0083 | 8.18 | coverage spread in chew_rate_hz; coverage spread in jaw_opening_mean; coverage spread in rhythm_variance; diverse asymmetry_band; diverse chew_rate_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 18 | 18 | 18 | 10.3282 | 1.0126 | 0.4613 | 691.8029 | 19.1116 | 0.0075 | 8.08 | coverage spread in chew_rate_hz; diverse asymmetry_band; diverse chew_rate_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 20 | 20 | 20 | 2.4592 | 1.0161 | 0.2514 | 72.1573 | 3.8991 | 0.0022 | 7.94 | diverse chew_rate_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 10 | 10 | 10 | 5.4107 | 1.0775 | 0.6200 | 310.4964 | 8.2758 | 0.0081 | 7.93 | coverage spread in asymmetry_ratio; coverage spread in pause_ratio; diverse pause_style_band; diverse rhythm_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 13 | 13 | 13 | 8.7828 | 0.9810 | 0.3007 | 231.7661 | 15.4141 | 0.0110 | 7.91 | coverage spread in chew_rate_hz; coverage spread in dominant_frequency; diverse asymmetry_band; diverse chew_rate_band; diverse rhythm_band; missing chew_side_label; missing label_quality; unlabeled food_type |
| 4 | 4 | 4 | 2.9775 | 0.9574 | 0.2743 | 415.1520 | 15.0293 | 0.0035 | 7.91 | diverse asymmetry_band; diverse chew_rate_band; diverse rhythm_band; missing chew_side_label; missing label_quality; unlabeled food_type |