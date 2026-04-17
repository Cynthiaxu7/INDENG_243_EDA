#!/usr/bin/env python3
"""Turn synthetic training artifacts into publication-quality figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from make_presentation_figures import PALETTE, save_dual  # type: ignore
    _REUSED_HELPERS = True
except Exception:
    _REUSED_HELPERS = False
    PALETTE = {
        "primary": "#2563EB",
        "secondary": "#0EA5E9",
        "teal": "#14B8A6",
        "orange": "#F59E0B",
        "magenta": "#D946EF",
        "red": "#EF4444",
        "slate": "#475569",
        "green": "#10B981",
    }

    def save_dual(fig: plt.Figure, out_dir: Path, stem: str) -> Tuple[Path, Path]:
        png = out_dir / f"{stem}.png"
        svg = out_dir / f"{stem}.svg"
        fig.savefig(png, dpi=320)
        fig.savefig(svg)
        plt.close(fig)
        return png, svg


NOISE_ORDER: List[str] = ["low", "med", "high"]
MODEL_ORDER: List[str] = ["logreg", "rf", "lgbm"]
CLASS_ORDER: List[str] = ["soft", "medium", "hard"]
MODEL_COLORS: Dict[str, str] = {
    "logreg": PALETTE["primary"],
    "rf": PALETTE["teal"],
    "lgbm": PALETTE["orange"],
}
CLASS_COLORS: Dict[str, str] = {
    "soft": PALETTE["green"],
    "medium": PALETTE["secondary"],
    "hard": PALETTE["red"],
}
PERTURBED_FEATURES = {
    "mouth_open_smooth_mean",
    "mouth_open_smooth_std",
    "mouth_open_smooth_p95",
    "mouth_open_px_mean",
    "mouth_open_px_std",
    "mouth_open_px_p95",
    "jaw_lr_delta_abs_mean",
    "pause_ratio",
}


def set_style() -> None:
    """Apply the shared seaborn/matplotlib style."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#E5E7EB",
            "grid.color": "#E5E7EB",
            "grid.alpha": 0.7,
            "axes.grid": True,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
        }
    )


def _require(path: Path) -> Path:
    """Raise FileNotFoundError with path if the artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def _subtitle(ax: plt.Axes, text: str) -> None:
    """Place a small muted subtitle above the axes."""
    ax.text(0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color="#64748B")


def _fig_subtitle(fig: plt.Figure, text: str, y: float = 0.95) -> None:
    """Add a centered subtitle to the figure under the suptitle."""
    fig.text(0.5, y, text, ha="center", va="top", fontsize=12, color="#64748B")


def _present_models(df: pd.DataFrame) -> List[str]:
    """Return models that are actually present, preserving preferred order."""
    present = set(df["model"].astype(str).unique())
    return [m for m in MODEL_ORDER if m in present]


def _fold_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric-fold rows (drop pre-aggregated mean rows)."""
    folds = df["fold"].astype(str).str.lower()
    return df.loc[folds != "mean"].copy()


def fig_f1_decay(cv_metrics: pd.DataFrame, out_dir: Path) -> str:
    """Macro-F1 decay curve across noise levels, one line per model."""
    df = _fold_rows(cv_metrics)
    df = df[df["noise_level"].isin(NOISE_ORDER)].copy()
    df["noise_level"] = pd.Categorical(df["noise_level"], categories=NOISE_ORDER, ordered=True)
    models = _present_models(df)

    agg = (
        df.groupby(["model", "noise_level"], observed=True)["macro_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 7))
    x_positions = np.arange(len(NOISE_ORDER))
    for model in models:
        sub = agg[agg["model"] == model].set_index("noise_level").reindex(NOISE_ORDER)
        means = sub["mean"].to_numpy()
        stds = sub["std"].to_numpy()
        color = MODEL_COLORS.get(model, PALETTE["slate"])
        ax.plot(x_positions, means, marker="o", linewidth=2.4, markersize=9, color=color, label=model)
        ax.fill_between(x_positions, means - stds, means + stds, color=color, alpha=0.18, linewidth=0)

    ax.axhline(1.0 / 3.0, linestyle="--", color=PALETTE["slate"], linewidth=1.6, alpha=0.8, label="random baseline")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(NOISE_ORDER)
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Macro-F1 Decay Under Injected Noise")
    _subtitle(ax, "Three-class problem (soft / medium / hard); shaded band = +/- 1 std across folds")
    ax.legend(loc="lower left", frameon=True)
    save_dual(fig, out_dir, "synthetic_01_f1_decay_curve")
    return "synthetic_01_f1_decay_curve"


def fig_confusion_matrices_grid(cm_data: dict, out_dir: Path) -> str:
    """Grid of normalized confusion matrices: rows=noise levels, cols=models."""
    available_models: List[str] = []
    for noise in NOISE_ORDER:
        for model in MODEL_ORDER:
            if model in cm_data.get(noise, {}) and model not in available_models:
                available_models.append(model)
    if not available_models:
        raise ValueError("No models found in synthetic_confusion_matrices.json")

    n_rows = len(NOISE_ORDER)
    n_cols = len(available_models)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(11, 4.0 * n_cols + 2), 3.8 * n_rows + 1.2),
        squeeze=False,
    )
    fig.suptitle("Confusion Matrices by Noise Level x Model", fontsize=20, y=1.0)

    for r, noise in enumerate(NOISE_ORDER):
        for c, model in enumerate(available_models):
            ax = axes[r][c]
            entry = cm_data.get(noise, {}).get(model)
            if not entry:
                ax.axis("off")
                ax.set_title(f"{model} - {noise}\n(missing)", fontsize=11, color="#94A3B8")
                continue
            labels = entry.get("labels", CLASS_ORDER)
            matrix = np.asarray(entry.get("normalized") or entry.get("matrix"), dtype=float)
            if entry.get("normalized") is None and matrix.sum() > 0:
                row_sums = matrix.sum(axis=1, keepdims=True)
                matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

            sns.heatmap(
                matrix,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                vmin=0.0,
                vmax=1.0,
                cbar=(c == n_cols - 1),
                cbar_kws={"shrink": 0.75} if c == n_cols - 1 else None,
                linewidths=0.4,
                linecolor="#E2E8F0",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
            )
            if r == 0:
                ax.set_title(model, fontsize=13)
            else:
                ax.set_title("")
            if c == 0:
                ax.set_ylabel(f"{noise}\nTrue", fontsize=11)
            else:
                ax.set_ylabel("")
            if r == n_rows - 1:
                ax.set_xlabel("Predicted", fontsize=11)
            else:
                ax.set_xlabel("")

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    save_dual(fig, out_dir, "synthetic_02_confusion_matrices_grid")
    return "synthetic_02_confusion_matrices_grid"


def fig_feature_importance_low(importance: pd.DataFrame, out_dir: Path) -> str:
    """Top-15 feature importance bars for each model at low noise."""
    df = importance[importance["noise_level"] == "low"].copy()
    if df.empty:
        raise ValueError("No low-noise rows in synthetic_feature_importance.csv")
    models = _present_models(df)

    n = len(models)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(max(12, 5.5 * n + 1), 8.5), squeeze=False)
    fig.suptitle("Top Feature Importances (Low Noise)", fontsize=20, y=1.02)
    _fig_subtitle(fig, "Perturbed features highlighted; all others shown in slate.", y=0.965)

    for ax, model in zip(axes[0], models):
        sub = df[df["model"] == model].copy()
        sub = sub.sort_values("importance", ascending=False).head(15)
        sub = sub.sort_values("importance", ascending=True)
        colors = [
            PALETTE["primary"] if feat in PERTURBED_FEATURES else PALETTE["slate"]
            for feat in sub["feature"]
        ]
        ax.barh(sub["feature"], sub["importance"], color=colors, edgecolor="white")
        ax.set_title(model, fontsize=13)
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=10)

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=PALETTE["primary"], label="Perturbed feature"),
        Patch(facecolor=PALETTE["slate"], label="Other feature"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_dual(fig, out_dir, "synthetic_03_feature_importance_low")
    return "synthetic_03_feature_importance_low"


def fig_per_class_f1(cv_metrics: pd.DataFrame, out_dir: Path) -> str:
    """Per-class F1 grouped bars by noise level, one panel per model."""
    df = _fold_rows(cv_metrics)
    df = df[df["noise_level"].isin(NOISE_ORDER)].copy()
    models = _present_models(df)

    long_rows = []
    for model in models:
        for noise in NOISE_ORDER:
            chunk = df[(df["model"] == model) & (df["noise_level"] == noise)]
            for cls in CLASS_ORDER:
                col = f"{cls}_f1"
                if col in chunk.columns and not chunk[col].dropna().empty:
                    vals = chunk[col].dropna().to_numpy()
                    long_rows.append(
                        {
                            "model": model,
                            "noise_level": noise,
                            "class": cls,
                            "f1_mean": float(np.mean(vals)),
                            "f1_std": float(np.std(vals, ddof=0)),
                        }
                    )
    long = pd.DataFrame(long_rows)

    n = len(models)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(max(12, 5.5 * n + 1), 7), sharey=True, squeeze=False)
    fig.suptitle("Per-Class F1 by Noise Level", fontsize=20, y=1.0)

    x = np.arange(len(NOISE_ORDER))
    bar_w = 0.26
    for ax, model in zip(axes[0], models):
        sub = long[long["model"] == model]
        for i, cls in enumerate(CLASS_ORDER):
            means = []
            stds = []
            for noise in NOISE_ORDER:
                row = sub[(sub["noise_level"] == noise) & (sub["class"] == cls)]
                if row.empty:
                    means.append(np.nan)
                    stds.append(0.0)
                else:
                    means.append(float(row["f1_mean"].iloc[0]))
                    stds.append(float(row["f1_std"].iloc[0]))
            offsets = x + (i - 1) * bar_w
            ax.bar(
                offsets,
                means,
                width=bar_w,
                yerr=stds,
                capsize=3,
                color=CLASS_COLORS[cls],
                edgecolor="white",
                label=cls,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(NOISE_ORDER)
        ax.set_title(model, fontsize=13)
        ax.set_xlabel("Noise Level")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("F1")

    axes[0][-1].legend(title="Class", loc="upper right", frameon=True)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_dual(fig, out_dir, "synthetic_04_per_class_f1_by_noise")
    return "synthetic_04_per_class_f1_by_noise"


def fig_probability_calibration(oof: pd.DataFrame, out_dir: Path) -> str:
    """Histogram of predicted probability for the TRUE class, per model (low noise)."""
    df = oof[oof["noise_level"] == "low"].copy()
    if df.empty:
        raise ValueError("No low-noise rows in synthetic_oof_predictions.csv")
    models = _present_models(df)

    prob_cols = {cls: f"prob_{cls}" for cls in CLASS_ORDER}
    missing = [c for c in prob_cols.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns in OOF predictions: {missing}")

    n = len(models)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(max(11, 5.0 * n + 1), 6.5), sharey=True, squeeze=False)
    fig.suptitle("Predicted Probability of True Class (Low Noise)", fontsize=20, y=1.0)
    _fig_subtitle(fig, "Each histogram shows p(true class) given the correct label, stratified by class.", y=0.955)

    for ax, model in zip(axes[0], models):
        sub = df[df["model"] == model]
        for cls in CLASS_ORDER:
            cls_rows = sub[sub["y_true"].astype(str) == cls]
            if cls_rows.empty:
                continue
            probs = pd.to_numeric(cls_rows[prob_cols[cls]], errors="coerce").dropna().to_numpy()
            if probs.size == 0:
                continue
            ax.hist(
                probs,
                bins=30,
                range=(0.0, 1.0),
                histtype="step",
                linewidth=2.2,
                color=CLASS_COLORS[cls],
                label=cls,
            )
        ax.set_title(model, fontsize=13)
        ax.set_xlabel("p(true class)")
        ax.set_ylabel("Count")
        ax.set_xlim(0.0, 1.0)
        ax.legend(title="True class", loc="upper left", frameon=True)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save_dual(fig, out_dir, "synthetic_05_probability_calibration")
    return "synthetic_05_probability_calibration"


def _load_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV or raise FileNotFoundError."""
    _require(path)
    return pd.read_csv(path)


def _load_json(path: Path) -> dict:
    """Read a required JSON or raise FileNotFoundError."""
    _require(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Render all synthetic-training figures into reports/figures/."""
    parser = argparse.ArgumentParser(description="Render synthetic training figures.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    if "MPLCONFIGDIR" in os.environ:
        matplotlib.rcParams["path.simplify"] = matplotlib.rcParams["path.simplify"]

    root: Path = args.project_root.resolve()
    out_dir: Path = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    results_dir = root / "artifacts" / "results"
    cv_metrics = _load_csv(results_dir / "synthetic_cv_metrics.csv")
    oof = _load_csv(results_dir / "synthetic_oof_predictions.csv")
    importance = _load_csv(results_dir / "synthetic_feature_importance.csv")
    cm_data = _load_json(results_dir / "synthetic_confusion_matrices.json")
    _ = _load_json(results_dir / "synthetic_run_log.json")

    produced: List[str] = []
    produced.append(fig_f1_decay(cv_metrics, out_dir))
    produced.append(fig_confusion_matrices_grid(cm_data, out_dir))
    produced.append(fig_feature_importance_low(importance, out_dir))
    produced.append(fig_per_class_f1(cv_metrics, out_dir))
    produced.append(fig_probability_calibration(oof, out_dir))

    print(f"Output directory: {out_dir}")
    print(f"Reused PALETTE/save_dual from make_presentation_figures: {_REUSED_HELPERS}")
    for stem in produced:
        print(f" - generated: {stem}.png / {stem}.svg")


if __name__ == "__main__":
    main()
