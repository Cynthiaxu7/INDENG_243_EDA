from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "eda_outputs"

plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
if HAS_SEABORN:
    sns.set_theme(style="whitegrid")


def safe_savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def get_dataset_dirs() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])


def write_basic_tables(df: pd.DataFrame, out_tables: Path, stem: str) -> None:
    out_tables.mkdir(parents=True, exist_ok=True)
    df.dtypes.astype(str).rename("dtype").to_csv(out_tables / f"{stem}_dtypes.csv")

    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_pct"] = (missing["missing_count"] / max(len(df), 1)) * 100
    missing.sort_values("missing_pct", ascending=False).to_csv(
        out_tables / f"{stem}_missingness.csv"
    )

    with open(out_tables / f"{stem}_shape.json", "w", encoding="utf-8") as f:
        json.dump({"rows": int(df.shape[0]), "columns": int(df.shape[1])}, f, indent=2)

    desc = df.describe(include="all").transpose()
    desc.to_csv(out_tables / f"{stem}_describe_all.csv")

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        numeric.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).transpose().to_csv(
            out_tables / f"{stem}_describe_numeric.csv"
        )


def plot_missingness(df: pd.DataFrame, out_figs: Path, stem: str) -> None:
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    top = missing_pct.head(30)
    plt.figure(figsize=(10, 4))
    top.plot(kind="bar", color="#5A7D9A")
    plt.title(f"{stem}: Missingness (%) - Top 30 Columns")
    plt.ylabel("Percent missing")
    plt.xlabel("Column")
    plt.xticks(rotation=70, ha="right")
    safe_savefig(out_figs / f"{stem}_missingness_bar.png")


def plot_numeric_histograms(df: pd.DataFrame, out_figs: Path, stem: str) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return
    max_cols = min(len(num_cols), 12)
    cols = num_cols[:max_cols]
    n_cols = 3
    n_rows = int(np.ceil(max_cols / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        ax = axes[i]
        series = df[col].dropna()
        ax.hist(series, bins=40, color="#6AA56A", alpha=0.9)
        ax.set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{stem}: Numeric Distributions (first {max_cols} columns)", y=1.02)
    safe_savefig(out_figs / f"{stem}_numeric_histograms.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_figs: Path, stem: str) -> None:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(corr, cmap="coolwarm", center=0, square=False, cbar=True)
    else:
        plt.imshow(corr.values, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
    plt.title(f"{stem}: Correlation Heatmap (numeric)")
    safe_savefig(out_figs / f"{stem}_correlation_heatmap.png")


def plot_mouth_timeseries(df: pd.DataFrame, out_figs: Path) -> None:
    needed = {"time_sec", "mouth_open_px", "mouth_width_px"}
    if not needed.issubset(df.columns):
        return
    tmp = df.sort_values("time_sec").copy()
    tmp["mouth_open_roll"] = tmp["mouth_open_px"].rolling(window=45, min_periods=1).mean()

    plt.figure(figsize=(12, 4))
    plt.plot(tmp["time_sec"], tmp["mouth_open_px"], alpha=0.35, label="mouth_open_px")
    plt.plot(tmp["time_sec"], tmp["mouth_open_roll"], linewidth=2, label="rolling mean (45)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Mouth open (px)")
    plt.title("Mouth Opening Over Time")
    plt.legend()
    safe_savefig(out_figs / "mouth_timeseries_open_over_time.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(
        tmp["mouth_width_px"],
        tmp["mouth_open_px"],
        s=8,
        alpha=0.25,
        c=tmp["time_sec"],
        cmap="viridis",
    )
    plt.xlabel("Mouth width (px)")
    plt.ylabel("Mouth open (px)")
    plt.title("Mouth Width vs Opening")
    plt.colorbar(label="Time (sec)")
    safe_savefig(out_figs / "mouth_timeseries_width_vs_open.png")


def plot_mouth_metrics(df: pd.DataFrame, out_figs: Path) -> None:
    needed = {"time_sec", "mouth_open_smooth", "mouth_is_open"}
    if not needed.issubset(df.columns):
        return
    tmp = df.sort_values("time_sec")

    plt.figure(figsize=(12, 4))
    plt.plot(tmp["time_sec"], tmp["mouth_open_smooth"], label="mouth_open_smooth", linewidth=1.8)
    open_mask = tmp["mouth_is_open"] == 1
    plt.scatter(
        tmp.loc[open_mask, "time_sec"],
        tmp.loc[open_mask, "mouth_open_smooth"],
        s=8,
        alpha=0.5,
        label="mouth_is_open=1",
    )
    plt.xlabel("Time (sec)")
    plt.ylabel("Smoothed mouth opening")
    plt.title("Smoothed Mouth Opening and Open-State Events")
    plt.legend()
    safe_savefig(out_figs / "mouth_metrics_smooth_and_open_events.png")

    if "bite_id" in tmp.columns:
        bite_len = tmp.groupby("bite_id").size()
        plt.figure(figsize=(9, 4))
        bite_len.plot(kind="hist", bins=40, color="#B57FC8")
        plt.title("Distribution of Frames per Bite")
        plt.xlabel("Frames per bite_id")
        safe_savefig(out_figs / "mouth_metrics_bite_frame_count_distribution.png")


def plot_chewing_analysis(df: pd.DataFrame, out_figs: Path) -> None:
    if "duration_sec" in df.columns:
        plt.figure(figsize=(8, 4))
        plt.hist(df["duration_sec"].dropna(), bins=30, color="#D28445")
        plt.title("Chewing Segment Duration Distribution")
        plt.xlabel("Duration (sec)")
        safe_savefig(out_figs / "chewing_duration_distribution.png")

    needed = {"duration_sec", "chewing_frequency_per_sec"}
    if needed.issubset(df.columns):
        plt.figure(figsize=(7, 5))
        if HAS_SEABORN and "chew_side" in df.columns:
            sns.scatterplot(
                data=df,
                x="duration_sec",
                y="chewing_frequency_per_sec",
                hue="chew_side",
                s=45,
            )
        else:
            plt.scatter(df["duration_sec"], df["chewing_frequency_per_sec"], alpha=0.75)
        plt.title("Duration vs Chewing Frequency")
        plt.xlabel("Duration (sec)")
        plt.ylabel("Chewing frequency (/sec)")
        safe_savefig(out_figs / "chewing_duration_vs_frequency.png")

    if {"chew_side", "duration_sec"}.issubset(df.columns):
        plt.figure(figsize=(7, 4))
        if HAS_SEABORN:
            sns.boxplot(data=df, x="chew_side", y="duration_sec")
        else:
            groups = [g["duration_sec"].dropna() for _, g in df.groupby("chew_side")]
            labels = list(df["chew_side"].dropna().unique())
            plt.boxplot(groups, labels=labels)
        plt.title("Duration by Chew Side")
        safe_savefig(out_figs / "chewing_duration_by_side_boxplot.png")


def plot_lip_landmarks(df: pd.DataFrame, out_figs: Path) -> None:
    needed = {"time_sec", "landmark_name", "y_px"}
    if not needed.issubset(df.columns):
        return
    targets = {"upper_lip", "lower_lip", "mouth_left", "mouth_right", "chin"}
    tmp = df[df["landmark_name"].isin(targets)].copy()
    if tmp.empty:
        return
    plt.figure(figsize=(12, 5))
    if HAS_SEABORN:
        sns.lineplot(data=tmp, x="time_sec", y="y_px", hue="landmark_name", linewidth=1.2)
    else:
        for name, g in tmp.groupby("landmark_name"):
            plt.plot(g["time_sec"], g["y_px"], label=name, linewidth=1.2)
        plt.legend()
    plt.title("Selected Lip/Jaw Landmark Y Position Over Time")
    plt.xlabel("Time (sec)")
    plt.ylabel("y_px")
    safe_savefig(out_figs / "lip_landmarks_y_over_time.png")


def plot_face_landmarks(df: pd.DataFrame, out_figs: Path) -> None:
    needed = {"time_sec", "landmark", "x_px", "y_px"}
    if not needed.issubset(df.columns):
        return
    landmark_ids = {13, 14, 61, 291, 152, 93, 323}
    tmp = df[df["landmark"].isin(landmark_ids)].copy()
    if tmp.empty:
        return

    plt.figure(figsize=(12, 5))
    if HAS_SEABORN:
        sns.lineplot(data=tmp, x="time_sec", y="y_px", hue="landmark", linewidth=1.0)
    else:
        for lm, g in tmp.groupby("landmark"):
            plt.plot(g["time_sec"], g["y_px"], label=str(lm), linewidth=1.0)
        plt.legend(title="landmark")
    plt.title("Selected Face Landmark Y Position Over Time")
    plt.xlabel("Time (sec)")
    plt.ylabel("y_px")
    safe_savefig(out_figs / "face_landmarks_y_over_time.png")


def run_generic_eda(df: pd.DataFrame, dataset_out: Path, csv_stem: str) -> None:
    figs = dataset_out / "figures"
    tables = dataset_out / "tables"
    write_basic_tables(df, tables, csv_stem)
    plot_missingness(df, figs, csv_stem)
    plot_numeric_histograms(df, figs, csv_stem)
    plot_correlation_heatmap(df, figs, csv_stem)


def run_file_specific_plots(name: str, df: pd.DataFrame, dataset_out: Path) -> None:
    figs = dataset_out / "figures"
    if name == "mouth_timeseries.csv":
        plot_mouth_timeseries(df, figs)
    elif name == "mouth_metrics.csv":
        plot_mouth_metrics(df, figs)
    elif name == "chewing_analysis.csv":
        plot_chewing_analysis(df, figs)
    elif name == "lip_landmarks.csv":
        plot_lip_landmarks(df, figs)
    elif name == "face_landmarks.csv":
        plot_face_landmarks(df, figs)


def cross_dataset_comparison(dataframes_by_dataset: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    comp_figs = OUT_DIR / "_cross_dataset" / "figures"
    comp_tabs = OUT_DIR / "_cross_dataset" / "tables"
    comp_figs.mkdir(parents=True, exist_ok=True)
    comp_tabs.mkdir(parents=True, exist_ok=True)

    chewing_frames = []
    mouth_frames = []
    for dataset, files in dataframes_by_dataset.items():
        if "chewing_analysis.csv" in files:
            d = files["chewing_analysis.csv"].copy()
            d["dataset"] = dataset
            chewing_frames.append(d)
        if "mouth_timeseries.csv" in files:
            m = files["mouth_timeseries.csv"][["time_sec", "mouth_open_px"]].copy()
            m["dataset"] = dataset
            mouth_frames.append(m)

    if chewing_frames:
        chew_all = pd.concat(chewing_frames, ignore_index=True)
        chew_all.to_csv(comp_tabs / "chewing_analysis_all_datasets.csv", index=False)

        if {"dataset", "duration_sec"}.issubset(chew_all.columns):
            plt.figure(figsize=(9, 5))
            if HAS_SEABORN:
                sns.violinplot(data=chew_all, x="dataset", y="duration_sec", inner="quartile")
            else:
                groups = [g["duration_sec"].dropna() for _, g in chew_all.groupby("dataset")]
                labels = list(chew_all["dataset"].dropna().unique())
                plt.boxplot(groups, labels=labels)
            plt.title("Chewing Duration by Dataset")
            plt.xticks(rotation=20, ha="right")
            safe_savefig(comp_figs / "chewing_duration_by_dataset.png")

    if mouth_frames:
        mouth_all = pd.concat(mouth_frames, ignore_index=True)
        mouth_all.to_csv(comp_tabs / "mouth_open_all_datasets.csv", index=False)

        plt.figure(figsize=(9, 5))
        if HAS_SEABORN:
            sns.kdeplot(data=mouth_all, x="mouth_open_px", hue="dataset", fill=True, common_norm=False)
        else:
            for ds, g in mouth_all.groupby("dataset"):
                plt.hist(g["mouth_open_px"].dropna(), bins=60, alpha=0.35, label=ds, density=True)
            plt.legend()
        plt.title("Mouth Opening Distribution by Dataset")
        plt.xlabel("mouth_open_px")
        safe_savefig(comp_figs / "mouth_open_distribution_by_dataset.png")


def main() -> None:
    dataset_dirs = get_dataset_dirs()
    if not dataset_dirs:
        print(f"No dataset directories found under: {DATA_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataframes_by_dataset: Dict[str, Dict[str, pd.DataFrame]] = {}

    for ds_dir in dataset_dirs:
        dataset_name = ds_dir.name
        ds_out = OUT_DIR / dataset_name
        ds_out.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(ds_dir.glob("*.csv"))
        dataframes_by_dataset[dataset_name] = {}
        run_log = {
            "dataset": dataset_name,
            "csv_files": [f.name for f in csv_files],
            "file_count": len(csv_files),
        }

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception as exc:
                print(f"Failed reading {csv_file}: {exc}")
                continue

            dataframes_by_dataset[dataset_name][csv_file.name] = df
            run_generic_eda(df, ds_out, csv_file.stem)
            run_file_specific_plots(csv_file.name, df, ds_out)

        with open(ds_out / "run_log.json", "w", encoding="utf-8") as f:
            json.dump(run_log, f, indent=2)

    cross_dataset_comparison(dataframes_by_dataset)
    print(f"EDA complete. Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
