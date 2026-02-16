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
