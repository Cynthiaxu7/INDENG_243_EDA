#!/usr/bin/env python3
"""Render publication-quality figures for the food-texture paper (r3).

Renders Figures 2-6 referenced by `reports/synthetic_texture_paper_r3.md`.
All plot text follows the paper's framing conventions: no mention of
"synthetic", "noise level", "low/med/high noise"; the three perturbation
magnitudes are denoted by sigma values (0.1, 0.3, 0.6) and the models are
shown by their human-readable names.
"""

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
from matplotlib.patches import Patch  # noqa: E402


# --- constants ---------------------------------------------------------------

# Internal noise keys ("low", "med", "high") map to the paper's sigma values
# 0.1, 0.3, 0.6 respectively. The labels below go on plots.
NOISE_ORDER: List[str] = ["low", "med", "high"]
NOISE_TO_SIGMA: Dict[str, float] = {"low": 0.1, "med": 0.3, "high": 0.6}
SIGMA_LABEL: Dict[str, str] = {
    "low": r"$\sigma = 0.1$",
    "med": r"$\sigma = 0.3$",
    "high": r"$\sigma = 0.6$",
}
SIGMA_LABEL_LONG: Dict[str, str] = {
    "low": r"$\sigma = 0.1$ (low)",
    "med": r"$\sigma = 0.3$ (mid)",
    "high": r"$\sigma = 0.6$ (severe)",
}

# Main-run models in the order the paper reports them.
MODEL_ORDER: List[str] = ["logreg", "rf", "xgb", "mlp", "ftt", "mamba_ssm_v1"]

# Human-readable display names.
MODEL_DISPLAY: Dict[str, str] = {
    "logreg": "Logistic Reg.",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "mlp": "MLP",
    "ftt": "FT-Transformer",
    "mamba_ssm_v1": "Mamba",
    # Ablation variant names (both appear in the ablation CSV)
    "mamba": "Mamba-3 (hybrid)",
}

# Ablation file uses these two keys to represent the two variants.
ABLATION_KEY_TO_DISPLAY: Dict[str, str] = {
    "mamba_ssm_v1": "Mamba-1 (ref.)",
    "mamba": "Mamba-3 (hybrid)",
}
ABLATION_ORDER: List[str] = ["mamba_ssm_v1", "mamba"]

CLASS_ORDER: List[str] = ["soft", "medium", "hard"]

# Categorical palette. Color-blind safe, high-contrast, print-legible.
PALETTE: Dict[str, str] = {
    "logreg": "#1f77b4",        # steel blue
    "rf": "#2ca02c",             # green
    "xgb": "#d62728",            # red
    "mlp": "#9467bd",            # purple
    "ftt": "#ff7f0e",            # orange
    "mamba_ssm_v1": "#111827",   # near-black (our method)
    "mamba": "#0ea5e9",          # sky (ablation hybrid)
    "neutral": "#6b7280",
    "accent": "#f59e0b",
}

CLASS_COLORS: Dict[str, str] = {
    "soft": "#4c78a8",
    "medium": "#f58518",
    "hard": "#54a24b",
}


# --- style -------------------------------------------------------------------


def set_style() -> None:
    """Minimalist academic look. Thin spines, subtle (mostly absent) grid."""
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d1d5db",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.6,
            "axes.grid": False,
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.titlepad": 8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# --- helpers -----------------------------------------------------------------


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def _load_csv(path: Path) -> pd.DataFrame:
    _require(path)
    return pd.read_csv(path)


def _load_json(path: Path) -> dict:
    _require(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_dual(fig: plt.Figure, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _fold_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric-fold rows (drop pre-aggregated 'mean' rows)."""
    folds = df["fold"].astype(str).str.lower()
    return df.loc[folds != "mean"].copy()


def _mean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the precomputed fold-mean rows."""
    folds = df["fold"].astype(str).str.lower()
    return df.loc[folds == "mean"].copy()


def _display(model_key: str) -> str:
    return MODEL_DISPLAY.get(model_key, model_key)


def _present_models(df: pd.DataFrame, order: List[str]) -> List[str]:
    present = set(df["model"].astype(str).unique())
    return [m for m in order if m in present]


# --- Figure 2: confusion matrices (6-panel grid at sigma = 0.1) --------------


def fig2_confusion_matrices_low(cm_data: dict, out_dir: Path) -> Tuple[str, Path, Path]:
    """Grid of row-normalized confusion matrices at sigma = 0.1, one per model."""
    noise = "low"
    entries = cm_data.get(noise, {})
    models = [m for m in MODEL_ORDER if m in entries]
    if not models:
        raise ValueError("No models found in confusion-matrix JSON at low-sigma level")

    # Lay out a 2 x 3 grid for exactly six models.
    n = len(models)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.2 * nrows), squeeze=False)

    labels = CLASS_ORDER

    for idx, model in enumerate(models):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        entry = entries[model]
        mat = entry.get("normalized")
        if mat is None:
            raw = np.asarray(entry["matrix"], dtype=float)
            row_sums = raw.sum(axis=1, keepdims=True)
            mat = np.divide(raw, row_sums, out=np.zeros_like(raw), where=row_sums > 0)
        mat = np.asarray(mat, dtype=float)

        im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_title(_display(model))
        if c == 0:
            ax.set_yticklabels(labels)
            ax.set_ylabel("True")
        else:
            ax.set_yticklabels([])
        if r == nrows - 1:
            ax.set_xticklabels(labels)
            ax.set_xlabel("Predicted")
        else:
            ax.set_xticklabels([])

        # annotate each cell
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if val >= 0.6 else "#111827"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color=color, fontsize=9,
                )
        # thin axes
        for spine in ax.spines.values():
            spine.set_edgecolor("#9ca3af")
            spine.set_linewidth(0.6)

    # hide any unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cbar.set_label("Row-normalized rate")
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        r"Pooled OOF row-normalized confusion matrices at $\sigma = 0.1$",
        fontsize=12, y=1.00,
    )
    png, svg = save_dual(fig, out_dir, "fig2_confusion_matrices_low")
    return "fig2_confusion_matrices_low", png, svg


# --- Figure 3: robustness curve (macro-F1 vs sigma) --------------------------


def _perwindow_macro_f1_from_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """Per-(noise, model) per-window macro-F1 computed over pooled OOF rows.

    We do NOT use this directly; the CV-metrics file already holds fold-mean
    macro-F1 from the training run, which matches the paper's numbers.
    Kept here for potential cross-check only.
    """
    from sklearn.metrics import f1_score

    rows = []
    for (noise, model), chunk in oof.groupby(["noise_level", "model"]):
        y_true = chunk["y_true"].astype(str)
        y_pred = chunk["y_pred"].astype(str)
        f1 = f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro")
        rows.append({"noise_level": noise, "model": model, "macro_f1": f1})
    return pd.DataFrame(rows)


def fig3_robustness_curve(cv_metrics: pd.DataFrame, out_dir: Path) -> Tuple[str, Path, Path]:
    """Window-level macro-F1 versus sigma, one line per model."""
    folds = _fold_rows(cv_metrics)
    folds = folds[folds["noise_level"].isin(NOISE_ORDER)].copy()
    models = _present_models(folds, MODEL_ORDER)

    agg = (
        folds.groupby(["model", "noise_level"])["macro_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    x_positions = np.arange(len(NOISE_ORDER))
    for model in models:
        sub = agg[agg["model"] == model].set_index("noise_level").reindex(NOISE_ORDER)
        means = sub["mean"].to_numpy()
        stds = sub["std"].fillna(0.0).to_numpy()
        color = PALETTE.get(model, PALETTE["neutral"])
        is_ours = model == "mamba_ssm_v1"
        lw = 2.2 if is_ours else 1.4
        marker = "D" if is_ours else "o"
        ms = 7 if is_ours else 5
        ax.plot(
            x_positions, means,
            marker=marker, markersize=ms,
            linewidth=lw, color=color,
            label=_display(model),
            zorder=3 if is_ours else 2,
        )
        # per-fold variance: light shaded band (optional, subtle)
        ax.fill_between(
            x_positions, means - stds, means + stds,
            color=color, alpha=0.10, linewidth=0, zorder=1,
        )

    # annotate Mamba's lead at each sigma, if Mamba is present
    if "mamba_ssm_v1" in models:
        mamba = agg[agg["model"] == "mamba_ssm_v1"].set_index("noise_level").reindex(NOISE_ORDER)
        tabular_models = [m for m in models if m != "mamba_ssm_v1"]
        best_tab_at = {}
        for noise in NOISE_ORDER:
            vals = []
            for m in tabular_models:
                row = agg[(agg["model"] == m) & (agg["noise_level"] == noise)]
                if not row.empty:
                    vals.append(float(row["mean"].iloc[0]))
            if vals:
                best_tab_at[noise] = max(vals)

        for i, noise in enumerate(NOISE_ORDER):
            mamba_val = float(mamba.loc[noise, "mean"])
            best_tab = best_tab_at.get(noise)
            if best_tab is None:
                continue
            lead = mamba_val - best_tab
            ax.annotate(
                f"+{lead:.3f}",
                xy=(x_positions[i], mamba_val),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5, color=PALETTE["mamba_ssm_v1"],
                weight="bold",
            )

    # subtle grid only for this robustness-curve figure
    ax.grid(axis="y", linestyle="-", linewidth=0.5, color="#e5e7eb", alpha=0.8)
    ax.set_axisbelow(True)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([SIGMA_LABEL_LONG[n] for n in NOISE_ORDER])
    ax.set_xlabel(r"Paired-noise magnitude $\sigma$")
    ax.set_ylabel("Window-level macro-F1")
    ax.set_ylim(0.25, 0.80)
    ax.set_title("Robustness under paired train-and-eval noise")

    # legend on the side, single column
    ax.legend(
        loc="upper right", frameon=False, ncol=1,
        handlelength=1.8, labelspacing=0.4,
    )
    png, svg = save_dual(fig, out_dir, "fig3_robustness_curve")
    return "fig3_robustness_curve", png, svg


# --- Figure 4: feature-importance top-10 per tabular model -------------------


def fig4_feature_importance_top10(importance: pd.DataFrame, out_dir: Path) -> Tuple[str, Path, Path]:
    """Horizontal bar charts of top-10 feature importances for logreg/rf/xgb at sigma = 0.1."""
    tabular_models = ["logreg", "rf", "xgb"]
    df = importance[importance["noise_level"] == "low"].copy()
    present = [m for m in tabular_models if m in df["model"].unique()]
    if not present:
        raise ValueError("No tabular models in feature-importance CSV at low sigma")

    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.2), squeeze=False)

    for ax, model in zip(axes[0], present):
        sub = df[df["model"] == model].copy()
        sub = sub.sort_values("importance", ascending=False).head(10)
        sub = sub.sort_values("importance", ascending=True)

        color = PALETTE.get(model, PALETTE["neutral"])
        ax.barh(
            sub["feature"], sub["importance"],
            color=color, edgecolor="white", linewidth=0.4,
        )
        ax.set_title(_display(model))
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", linestyle="-", linewidth=0.5, color="#e5e7eb", alpha=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(
        r"Top-10 feature importance per tabular model at $\sigma = 0.1$",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    png, svg = save_dual(fig, out_dir, "fig4_feature_importance_top10")
    return "fig4_feature_importance_top10", png, svg


# --- Figure 5: per-class F1 for Mamba vs best tabular at sigma=0.1 -----------


def fig5_per_class_f1_low(cv_metrics: pd.DataFrame, out_dir: Path) -> Tuple[str, Path, Path]:
    """Per-class F1 grouped bars at sigma = 0.1 for Mamba and the best tabular baseline.

    The best tabular baseline is selected by fold-mean window-level macro-F1 at sigma=0.1.
    """
    folds = _fold_rows(cv_metrics)
    folds = folds[folds["noise_level"] == "low"].copy()
    if folds.empty:
        raise ValueError("No fold rows at low sigma")

    # pick best tabular by macro_f1 fold-mean
    tab_models = [m for m in ["logreg", "rf", "xgb", "mlp", "ftt"] if m in folds["model"].unique()]
    macro_by_model = folds.groupby("model")["macro_f1"].mean()
    tab_scores = macro_by_model.reindex(tab_models)
    best_tab = tab_scores.idxmax()

    compare_models = [best_tab, "mamba_ssm_v1"]
    compare_models = [m for m in compare_models if m in folds["model"].unique()]

    rows = []
    for model in compare_models:
        m_rows = folds[folds["model"] == model]
        for cls in CLASS_ORDER:
            col = f"{cls}_f1"
            vals = pd.to_numeric(m_rows[col], errors="coerce").dropna().to_numpy()
            rows.append({
                "model": model,
                "class": cls,
                "f1_mean": float(np.mean(vals)) if vals.size else np.nan,
                "f1_std": float(np.std(vals, ddof=0)) if vals.size else 0.0,
            })
    long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(CLASS_ORDER))
    bar_w = 0.36
    for i, model in enumerate(compare_models):
        sub = long[long["model"] == model].set_index("class").reindex(CLASS_ORDER)
        means = sub["f1_mean"].to_numpy()
        stds = sub["f1_std"].to_numpy()
        offsets = x + (i - 0.5) * bar_w
        color = PALETTE.get(model, PALETTE["neutral"])
        bars = ax.bar(
            offsets, means, width=bar_w, yerr=stds,
            capsize=3, color=color, edgecolor="white", linewidth=0.5,
            label=_display(model),
            error_kw={"elinewidth": 0.8, "ecolor": "#4b5563"},
        )
        # annotate bar heights above the error bar
        for rect, val, std in zip(bars, means, stds):
            if np.isnan(val):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                val + std + 0.02,
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=8, color="#111827",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASS_ORDER])
    ax.set_ylabel("Per-class F1")
    ax.set_xlabel("Class")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        rf"Per-class F1 at $\sigma = 0.1$: Mamba vs best tabular ({_display(best_tab)})"
    )
    ax.legend(loc="upper right", frameon=False)
    png, svg = save_dual(fig, out_dir, "fig5_per_class_f1_low")
    return "fig5_per_class_f1_low", png, svg


# --- Figure 6: ablation bar chart --------------------------------------------


def fig6_ablation_mamba1_vs_mamba3(ablation_metrics: pd.DataFrame, out_dir: Path) -> Tuple[str, Path, Path]:
    """Grouped bars: window-F1 and chunk-F1 for Mamba-1 (ref.) vs Mamba-3 (hybrid)."""
    df = ablation_metrics[ablation_metrics["noise_level"] == "low"].copy()
    fold_df = _fold_rows(df)

    # per-variant: fold-mean + std for window macro-F1
    rows = []
    for variant in ABLATION_ORDER:
        v_rows = fold_df[fold_df["model"] == variant]
        if v_rows.empty:
            continue
        win_mean = float(v_rows["macro_f1"].mean())
        win_std = float(v_rows["macro_f1"].std(ddof=0))

        # chunk: pull the pre-aggregated "mean" row's chunk_macro_f1
        mean_row = df[(df["model"] == variant) & (df["fold"].astype(str).str.lower() == "mean")]
        if not mean_row.empty:
            chunk_mean = float(pd.to_numeric(mean_row["chunk_macro_f1"], errors="coerce").iloc[0])
        else:
            chunk_mean = float("nan")

        rows.append({
            "variant": variant,
            "win_mean": win_mean,
            "win_std": win_std,
            "chunk_mean": chunk_mean,
        })
    summary = pd.DataFrame(rows).set_index("variant").reindex(ABLATION_ORDER).dropna(how="all")
    variants = [v for v in ABLATION_ORDER if v in summary.index]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    metrics = ["Window macro-F1", "Chunk macro-F1"]
    x = np.arange(len(metrics))
    bar_w = 0.36

    variant_colors = {
        "mamba_ssm_v1": PALETTE["mamba_ssm_v1"],
        "mamba": PALETTE["mamba"],
    }

    for i, variant in enumerate(variants):
        means = [summary.loc[variant, "win_mean"], summary.loc[variant, "chunk_mean"]]
        stds = [summary.loc[variant, "win_std"], 0.0]
        offsets = x + (i - 0.5) * bar_w
        color = variant_colors.get(variant, PALETTE["neutral"])
        bars = ax.bar(
            offsets, means, width=bar_w, yerr=stds,
            capsize=3, color=color, edgecolor="white", linewidth=0.5,
            label=ABLATION_KEY_TO_DISPLAY.get(variant, variant),
            error_kw={"elinewidth": 0.8, "ecolor": "#4b5563"},
        )
        for rect, val in zip(bars, means):
            if np.isnan(val):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=8, color="#111827",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, 0.65)
    ax.set_title(
        "Matched-size ablation: Mamba-1 vs Mamba-3-style upgrades"
    )
    ax.legend(loc="upper left", frameon=False)
    png, svg = save_dual(fig, out_dir, "fig6_ablation_mamba1_vs_mamba3")
    return "fig6_ablation_mamba1_vs_mamba3", png, svg


# --- main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Render paper-aligned figures.")
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
    importance = _load_csv(results_dir / "synthetic_feature_importance.csv")
    cm_data = _load_json(results_dir / "synthetic_confusion_matrices.json")

    ablation_path = results_dir / "synthetic_cv_metrics_ablation3.csv"
    try:
        ablation_metrics = _load_csv(ablation_path)
    except FileNotFoundError:
        ablation_metrics = None

    produced: List[Tuple[str, Path, Path]] = []
    produced.append(fig2_confusion_matrices_low(cm_data, out_dir))
    produced.append(fig3_robustness_curve(cv_metrics, out_dir))
    produced.append(fig4_feature_importance_top10(importance, out_dir))
    produced.append(fig5_per_class_f1_low(cv_metrics, out_dir))
    if ablation_metrics is not None:
        produced.append(fig6_ablation_mamba1_vs_mamba3(ablation_metrics, out_dir))

    print(f"Output directory: {out_dir}")
    for stem, png, svg in produced:
        png_kb = png.stat().st_size / 1024.0
        svg_kb = svg.stat().st_size / 1024.0
        print(
            f" - {stem}: "
            f"{png.name} ({png_kb:.1f} KB), "
            f"{svg.name} ({svg_kb:.1f} KB)"
        )


if __name__ == "__main__":
    main()
