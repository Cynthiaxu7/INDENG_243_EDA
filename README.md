# INDENG 243 - Chewing Behavior EDA

This project performs exploratory data analysis (EDA) on mouth/chewing metrics extracted from video-based landmark pipelines.

The workflow reads CSV datasets from `data/`, computes detailed summary tables, and generates publication-ready figures in `eda_outputs/`.

## Project Structure

- `data/` - raw and processed CSV inputs (ignored by git)
  - per-dataset folders: `dessert/`, `jiayi/`, `rameen/`, etc.
  - combined files: `_ALL_*.csv`
- `detailed_eda.py` - main EDA script
- `eda_outputs/` - generated tables/figures (tracked in git)
  - per-dataset folders
  - `_cross_dataset/` comparisons
  - `top5_important_figures/` curated presentation figures

## Environment Setup

Use Python 3.10+ (3.12 recommended).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run EDA

Run the full EDA pipeline:

```bash
MPLCONFIGDIR=".mplconfig" python detailed_eda.py
```

What the script generates for each available table:

- `*_shape.json`
- `*_dtypes.csv`
- `*_missingness.csv`
- `*_describe_all.csv`
- `*_describe_numeric.csv`
- figure set including missingness, histogram, and correlation plots

It also creates cross-dataset outputs under:

- `eda_outputs/_cross_dataset/figures/`
- `eda_outputs/_cross_dataset/tables/`

## Top 5 Figure Pack

Curated figures for reporting are in:

- `eda_outputs/top5_important_figures/`

See:

- `eda_outputs/top5_important_figures/README_top5.txt`

for interpretation notes for each figure.

## Git Notes

The `.gitignore` is configured to ignore:

- raw videos (`*.mov`)
- dataset folder (`data/`)
- virtual env folders (`.venv/`, `venv/`, `env/`)
- archive files (`*.zip`)

Generated figures in `eda_outputs/` are intentionally **not** ignored so they can be pushed to remote.
