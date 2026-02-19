Top 5 important EDA figures (curated)
=====================================

1) 01_cross_section_mouth_open_over_normalized_time_7datasets.png
-----------------------------------------------------------------
What this figure is:
- A cross-sectional line plot comparing all 7 datasets (dessert, jiaqi, jiayi, rameen, richie, steak, zoe).
- X-axis is normalized time (0 to 1), so each video is scaled to the same relative progress.
- Y-axis is median mouth opening in pixels within each time bin.

How to read it:
- Compare line levels to see which dataset generally has larger mouth openings.
- Compare line shapes to see whether opening activity is front-loaded, middle-heavy, or end-heavy.
- Look for curve crossings to identify where one dataset overtakes another during the chewing timeline.

Why it matters:
- It removes absolute video-length differences and allows fair trajectory comparison.
- It shows temporal chewing behavior, not just one summary statistic.
- It helps detect subject/food-specific dynamics over time.

2) 02_cross_section_chewing_duration_by_dataset.png
---------------------------------------------------
What this figure is:
- A boxplot of chewing duration (seconds) grouped by dataset.
- Each box shows median, interquartile range (IQR), whiskers, and potential outliers.

How to read it:
- Median line: central chewing duration for each dataset.
- Box height (IQR): variability within typical observations.
- Long whiskers/outliers: occasional unusually short or long chewing episodes.

Why it matters:
- It gives a robust side-by-side comparison of chewing pace and consistency.
- Datasets with higher medians imply generally longer chew episodes.
- Wide spread indicates heterogeneous chewing behavior within that dataset.

3) 03_cross_section_mouth_open_distribution_by_dataset.png
----------------------------------------------------------
What this figure is:
- A violin plot of mouth opening (pixels) by dataset.
- Shape width indicates density (where values are more common).
- Internal quartile markers summarize center and spread.

How to read it:
- Wider regions = more frequent mouth-open values.
- Multiple bulges can indicate multi-modal behavior (different chewing states).
- Compare median/quartiles across datasets to assess typical openness levels.

Why it matters:
- It captures the full distribution better than a boxplot alone.
- It highlights whether a dataset has stable vs. highly variable mouth opening.
- It is useful for spotting potential dataset bias or calibration differences.

4) 04_all_data_mouth_timeseries_correlation_heatmap.png
-------------------------------------------------------
What this figure is:
- A correlation heatmap (Pearson) for numeric variables in the combined (_ALL) mouth timeseries data.
- Color intensity shows strength/direction of linear relationships.

How to read it:
- Values near +1: strong positive relationship.
- Values near -1: strong negative relationship.
- Values near 0: weak linear relationship.
- Check rows/columns involving mouth_open_px, mouth_width_px, and jaw coordinates for key biomechanics links.

Why it matters:
- It quickly identifies redundant features and candidate predictors.
- It supports feature engineering and model simplification decisions.
- It provides a global dependency overview before deeper modeling.

5) 05_all_data_chewing_duration_vs_frequency.png
------------------------------------------------
What this figure is:
- A scatter plot across all data showing chewing duration (x) versus chewing frequency per second (y).
- Each point represents a chewing segment.

How to read it:
- Overall slope/trend shows whether longer chew durations associate with higher or lower frequency.
- Point concentration regions identify common behavior regimes.
- Outliers suggest unusual chewing events or possible data-quality checks.

Why it matters:
- It connects two clinically/behaviorally meaningful metrics directly.
- It helps evaluate trade-offs between speed and duration of chewing.
- It informs whether simple linear trends exist or if nonlinear modeling may be needed.