# INDENG 243 - Chewing Video Data EDA

This repository contains CSV exports from chewing video processing and a Python EDA script that profiles the data and generates figures.

## Project structure

- `data/` - source CSV datasets
  - `JAIYI_videos_csv_files/`
  - `Rameen_Chewing_Video_Files/`
- `detailed_eda.py` - automated EDA pipeline
- `eda_outputs/` - generated figures and summary tables

## What the EDA script generates

For each CSV file in each dataset directory:

- `tables/`
  - `<file>_dtypes.csv`
  - `<file>_missingness.csv`
  - `<file>_describe_all.csv`
  - `<file>_describe_numeric.csv`
- `figures/`
  - Missingness bar chart
  - Numeric histograms
  - Correlation heatmap
  - File-specific plots (time-series, chewing analysis, landmark trends)

Cross-dataset comparisons are saved under:

- `eda_outputs/_cross_dataset/figures/`
- `eda_outputs/_cross_dataset/tables/`

## Plot guide (what each plot set means)

### Generic plots generated for every CSV

- `<file>_missingness_bar.png`: Percent missing values by column (top columns), used to quickly spot data quality gaps.
- `<file>_numeric_histograms.png`: Distribution of numeric columns, used to detect skew, outliers, and unusual ranges.
- `<file>_correlation_heatmap.png`: Pairwise numeric correlation matrix, used to find strongly related or redundant features.

### File-specific plot sets

- `mouth_timeseries_*`
  - `mouth_timeseries_open_over_time.png`: Raw mouth opening over time with a rolling average trend line.
  - `mouth_timeseries_width_vs_open.png`: Scatter of mouth width vs mouth opening (colored by time) to inspect movement patterns and coupling.

- `mouth_metrics_*`
  - `mouth_metrics_smooth_and_open_events.png`: Smoothed mouth opening with highlighted frames where `mouth_is_open == 1`.
  - `mouth_metrics_bite_frame_count_distribution.png`: Histogram of frames per `bite_id`, showing bite duration consistency.

- `chewing_analysis_*`
  - `chewing_duration_distribution.png`: Distribution of chewing segment durations.
  - `chewing_duration_vs_frequency.png`: Relationship between segment duration and chewing frequency (optionally colored by `chew_side`).
  - `chewing_duration_by_side_boxplot.png`: Duration comparison across chewing sides (left/right), useful for asymmetry checks.

- `lip_landmarks_*`
  - `lip_landmarks_y_over_time.png`: Y-position trajectories for selected lip/jaw landmarks (`upper_lip`, `lower_lip`, `mouth_left`, `mouth_right`, `chin`) across time.

- `face_landmarks_*`
  - `face_landmarks_y_over_time.png`: Y-position trajectories for selected face landmarks (`13`, `14`, `61`, `291`, `152`, `93`, `323`) across time.

### Cross-dataset comparison plots

- `chewing_duration_by_dataset.png`: Distribution comparison of chewing durations between dataset folders.
- `mouth_open_distribution_by_dataset.png`: Comparison of mouth opening distributions between dataset folders.

## Setup

Create and activate a local environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run EDA

Run the script from the repository root:

```bash
MPLCONFIGDIR="$(pwd)/.mplconfig" MPLBACKEND=Agg python detailed_eda.py
```

When complete, outputs are written to:

```bash
eda_outputs/
```

## Notes

- `MPLBACKEND=Agg` ensures figure generation in non-interactive environments.
- `MPLCONFIGDIR` avoids matplotlib cache permission issues on some systems.
- If your environment has binary version conflicts (for example, NumPy/Pandas mismatch), recreate `.venv` and reinstall from `requirements.txt`.
