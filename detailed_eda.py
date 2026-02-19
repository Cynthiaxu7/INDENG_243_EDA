from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    csv_files: Dict[str, Path]


def ensure_dirs(base_dir: Path) -> Tuple[Path, Path]:
    fig_dir = base_dir / "figures"
    tab_dir = base_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, tab_dir


def clean_output_dir(base_dir: Path) -> None:
    if not base_dir.exists():
        return
    for child in base_dir.rglob("*"):
        if child.is_file():
            child.unlink()


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def write_core_tables(df: pd.DataFrame, stem: str, tables_dir: Path) -> None:
    shape = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    (tables_dir / f"{stem}_shape.json").write_text(json.dumps(shape, indent=2), encoding="utf-8")

    dtypes = (
        pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
        .sort_values("column")
        .reset_index(drop=True)
    )
    dtypes.to_csv(tables_dir / f"{stem}_dtypes.csv", index=False)

    missing = (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": df.isna().sum().values,
                "missing_pct": (100.0 * df.isna().mean().values).round(4),
            }
        )
        .sort_values(["missing_count", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )
    missing.to_csv(tables_dir / f"{stem}_missingness.csv", index=False)

    try:
        df.describe(include="all").transpose().to_csv(tables_dir / f"{stem}_describe_all.csv")
    except Exception:
        pass

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        numeric.describe().transpose().to_csv(tables_dir / f"{stem}_describe_numeric.csv")


def save_missingness_plot(df: pd.DataFrame, title: str, out_path: Path) -> None:
    missing_pct = 100.0 * df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(max(8, min(16, 0.45 * len(missing_pct))), 4.8))
    ax = sns.barplot(x=missing_pct.index, y=missing_pct.values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel("Missing (%)")
    ax.set_xlabel("Column")
    ax.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_numeric_histograms(df: pd.DataFrame, title: str, out_path: Path) -> None:
    numeric_cols = list(df.select_dtypes(include="number").columns)
    if not numeric_cols:
        return
    max_cols = min(len(numeric_cols), 24)
    numeric_cols = numeric_cols[:max_cols]

    n_cols = 4
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        s = df[col].dropna()
        if s.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            ax.hist(s, bins=35, color="#59A14F", alpha=0.85)
        ax.set_title(col, fontsize=9)

    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame, title: str, out_path: Path) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr(numeric_only=True)
    plt.figure(figsize=(max(6, 0.7 * corr.shape[1]), max(5, 0.6 * corr.shape[1])))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.4)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_mouth_timeseries_plots(df: pd.DataFrame, figures_dir: Path, stem: str) -> None:
    if "time_sec" in df.columns and "mouth_open_px" in df.columns:
        plot_df = df[["time_sec", "mouth_open_px"]].dropna().sort_values("time_sec")
        if not plot_df.empty:
            plt.figure(figsize=(10, 4.8))
            sns.lineplot(data=plot_df, x="time_sec", y="mouth_open_px", color="#E15759", linewidth=1)
            plt.title("Mouth Opening Over Time")
            plt.xlabel("Time (sec)")
            plt.ylabel("Mouth Open (px)")
            plt.tight_layout()
            plt.savefig(figures_dir / f"{stem}_open_over_time.png")
            plt.close()

    if {"mouth_width_px", "mouth_open_px"}.issubset(df.columns):
        plot_df = df[["mouth_width_px", "mouth_open_px"]].dropna()
        if not plot_df.empty:
            if len(plot_df) > 50000:
                plot_df = plot_df.sample(50000, random_state=42)
            plt.figure(figsize=(7, 5))
            sns.scatterplot(data=plot_df, x="mouth_width_px", y="mouth_open_px", s=10, alpha=0.25)
            plt.title("Mouth Width vs Mouth Open")
            plt.xlabel("Mouth Width (px)")
            plt.ylabel("Mouth Open (px)")
            plt.tight_layout()
            plt.savefig(figures_dir / f"{stem}_width_vs_open.png")
            plt.close()


def save_landmark_y_plot(df: pd.DataFrame, figures_dir: Path, stem: str) -> None:
    if not {"time_sec", "y_px", "landmark_name"}.issubset(df.columns):
        return
    top_landmarks = df["landmark_name"].value_counts().head(8).index.tolist()
    plot_df = df[df["landmark_name"].isin(top_landmarks)][["time_sec", "y_px", "landmark_name"]].dropna()
    if plot_df.empty:
        return
    if len(plot_df) > 80000:
        plot_df = plot_df.sample(80000, random_state=42)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=plot_df, x="time_sec", y="y_px", hue="landmark_name", estimator=None, linewidth=0.9)
    plt.title(f"{stem} Y Position Over Time")
    plt.xlabel("Time (sec)")
    plt.ylabel("Y (px)")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{stem}_y_over_time.png")
    plt.close()


def save_chewing_plots(df: pd.DataFrame, figures_dir: Path) -> None:
    if "duration_sec" in df.columns:
        s = df["duration_sec"].dropna()
        if not s.empty:
            plt.figure(figsize=(7, 4.8))
            sns.histplot(s, bins=30, kde=True, color="#F28E2B")
            plt.title("Chewing Duration Distribution")
            plt.xlabel("Duration (sec)")
            plt.tight_layout()
            plt.savefig(figures_dir / "chewing_duration_distribution.png")
            plt.close()

    if {"duration_sec", "chew_side"}.issubset(df.columns):
        plot_df = df[["duration_sec", "chew_side"]].dropna()
        if not plot_df.empty:
            plt.figure(figsize=(7, 4.8))
            sns.boxplot(data=plot_df, x="chew_side", y="duration_sec", hue=None)
            plt.title("Chewing Duration by Side")
            plt.xlabel("Chew Side")
            plt.ylabel("Duration (sec)")
            plt.tight_layout()
            plt.savefig(figures_dir / "chewing_duration_by_side_boxplot.png")
            plt.close()

    if {"duration_sec", "chewing_frequency_per_sec"}.issubset(df.columns):
        plot_df = df[["duration_sec", "chewing_frequency_per_sec"]].dropna()
        if not plot_df.empty:
            plt.figure(figsize=(7, 4.8))
            sns.scatterplot(
                data=plot_df,
                x="duration_sec",
                y="chewing_frequency_per_sec",
                s=30,
                alpha=0.75,
                color="#4E79A7",
            )
            plt.title("Chewing Duration vs Frequency")
            plt.xlabel("Duration (sec)")
            plt.ylabel("Frequency (per sec)")
            plt.tight_layout()
            plt.savefig(figures_dir / "chewing_duration_vs_frequency.png")
            plt.close()


def run_table_eda(df: pd.DataFrame, stem: str, figures_dir: Path, tables_dir: Path) -> None:
    write_core_tables(df=df, stem=stem, tables_dir=tables_dir)
    save_missingness_plot(df=df, title=f"{stem} Missingness", out_path=figures_dir / f"{stem}_missingness_bar.png")
    save_numeric_histograms(
        df=df,
        title=f"{stem} Numeric Histograms",
        out_path=figures_dir / f"{stem}_numeric_histograms.png",
    )
    save_correlation_heatmap(
        df=df,
        title=f"{stem} Correlation Heatmap",
        out_path=figures_dir / f"{stem}_correlation_heatmap.png",
    )

    if stem == "mouth_timeseries":
        save_mouth_timeseries_plots(df=df, figures_dir=figures_dir, stem=stem)
    if stem in {"lip_landmarks", "face_landmarks"}:
        save_landmark_y_plot(df=df, figures_dir=figures_dir, stem=stem)
    if stem == "chewing_analysis":
        save_chewing_plots(df=df, figures_dir=figures_dir)


def discover_datasets(data_dir: Path) -> List[DatasetBundle]:
    datasets: List[DatasetBundle] = []

    for subdir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        csv_files = {p.stem: p for p in sorted(subdir.glob("*.csv"))}
        if csv_files:
            datasets.append(DatasetBundle(name=subdir.name, csv_files=csv_files))

    top_level = {p.stem.removeprefix("_ALL_"): p for p in sorted(data_dir.glob("_ALL_*.csv"))}
    if top_level:
        datasets.append(DatasetBundle(name="_ALL", csv_files=top_level))

    return datasets


def cross_dataset_outputs(
    datasets: Iterable[DatasetBundle],
    out_root: Path,
) -> None:
    fig_dir, tab_dir = ensure_dirs(out_root)

    chewing_frames = []
    mouth_frames = []
    for ds in datasets:
        if ds.name == "_ALL":
            continue
        if "chewing_analysis" in ds.csv_files:
            d = safe_read_csv(ds.csv_files["chewing_analysis"])
            d["dataset"] = ds.name
            chewing_frames.append(d)
        if "mouth_metrics" in ds.csv_files:
            d = safe_read_csv(ds.csv_files["mouth_metrics"])
            d["dataset"] = ds.name
            mouth_frames.append(d)

    if chewing_frames:
        chewing = pd.concat(chewing_frames, ignore_index=True)
        chewing.to_csv(tab_dir / "chewing_analysis_all_datasets.csv", index=False)
        if {"dataset", "duration_sec"}.issubset(chewing.columns):
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=chewing.dropna(subset=["duration_sec"]), x="dataset", y="duration_sec")
            plt.title("Chewing Duration by Dataset")
            plt.xlabel("Dataset")
            plt.ylabel("Duration (sec)")
            plt.tight_layout()
            plt.savefig(fig_dir / "chewing_duration_by_dataset.png")
            plt.close()

    if mouth_frames:
        mouth = pd.concat(mouth_frames, ignore_index=True)
        mouth.to_csv(tab_dir / "mouth_open_all_datasets.csv", index=False)
        if {"dataset", "mouth_open_px"}.issubset(mouth.columns):
            plot_df = mouth.dropna(subset=["mouth_open_px", "dataset"])
            if len(plot_df) > 100000:
                plot_df = plot_df.groupby("dataset", group_keys=False).apply(
                    lambda x: x.sample(min(len(x), 15000), random_state=42)
                )
            plt.figure(figsize=(10, 5))
            sns.violinplot(data=plot_df, x="dataset", y="mouth_open_px", cut=0, inner="quartile")
            plt.title("Mouth Open Distribution by Dataset")
            plt.xlabel("Dataset")
            plt.ylabel("Mouth Open (px)")
            plt.tight_layout()
            plt.savefig(fig_dir / "mouth_open_distribution_by_dataset.png")
            plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    out_root = project_root / "eda_outputs"

    datasets = discover_datasets(data_dir)
    if not datasets:
        raise RuntimeError(f"No datasets found under: {data_dir}")

    run_log: Dict[str, dict] = {"datasets": {}, "project_root": str(project_root)}

    for ds in datasets:
        ds_out = out_root / ds.name
        clean_output_dir(ds_out)
        figures_dir, tables_dir = ensure_dirs(ds_out)

        run_log["datasets"][ds.name] = {"tables": list(ds.csv_files.keys()), "outputs": []}
        print(f"Running EDA for dataset: {ds.name}")
        for stem, csv_path in sorted(ds.csv_files.items()):
            print(f"  - {stem}: {csv_path.name}")
            df = safe_read_csv(csv_path)
            run_table_eda(df=df, stem=stem, figures_dir=figures_dir, tables_dir=tables_dir)
            run_log["datasets"][ds.name]["outputs"].append(stem)

        (ds_out / "run_log.json").write_text(json.dumps(run_log["datasets"][ds.name], indent=2), encoding="utf-8")

    cross_out = out_root / "_cross_dataset"
    clean_output_dir(cross_out)
    cross_dataset_outputs(datasets=datasets, out_root=cross_out)

    (out_root / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    print(f"\nEDA complete. Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
