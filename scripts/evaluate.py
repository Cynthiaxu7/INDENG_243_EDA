#!/usr/bin/env python3
"""Aggregate grouped-CV results with video-level reporting safeguards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _score_column(row: pd.Series) -> float:
    if "metric_macro_f1" in row and pd.notna(row["metric_macro_f1"]):
        return float(row["metric_macro_f1"])
    if "metric_mae" in row and pd.notna(row["metric_mae"]):
        return -float(row["metric_mae"])
    return -1e9


def derive_severity(pred_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()

    prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
    if prob_cols:
        pred_df["uncertainty"] = 1.0 - pred_df[prob_cols].max(axis=1)
    else:
        pred_df["uncertainty"] = np.nan

    candidate_feat = [
        "chew_interval_cv",
        "pause_ratio",
        "chew_side_balance_abs_diff",
        "jaw_opening_amplitude_std",
        "rhythm_variance_from_smooth",
    ]
    have = [c for c in candidate_feat if c in feat_df.columns]
    if have:
        z = (feat_df[have] - feat_df[have].mean()) / feat_df[have].std(ddof=0).replace(0, np.nan)
        feat_df = feat_df.copy()
        feat_df["feature_deviation"] = z.abs().mean(axis=1).fillna(0.0)
        dev = feat_df[["sample_id", "feature_deviation"]]
        pred_df = pred_df.merge(dev, on="sample_id", how="left")
    else:
        pred_df["feature_deviation"] = np.nan

    # Severity derived from confidence + normalized deviations (not learned directly).
    u = pred_df["uncertainty"].fillna(pred_df["uncertainty"].median() if pred_df["uncertainty"].notna().any() else 0.5)
    d = pred_df["feature_deviation"].fillna(
        pred_df["feature_deviation"].median() if pred_df["feature_deviation"].notna().any() else 0.0
    )
    u_n = (u - u.min()) / (u.max() - u.min() + 1e-9)
    d_n = (d - d.min()) / (d.max() - d.min() + 1e-9)
    pred_df["severity_score_0_1"] = 0.6 * u_n + 0.4 * d_n
    return pred_df


def write_report(
    out_md: Path,
    metrics: pd.DataFrame,
    severity_df: pd.DataFrame,
    notes: List[str],
    recommendations: List[str],
) -> None:
    lines = [
        "# Baseline Report",
        "",
        "## Cross-Validation Summary",
        "",
    ]
    if metrics.empty:
        lines.append("- No metrics found. Train models first.")
    else:
        for _, r in metrics.iterrows():
            if pd.notna(r.get("metric_macro_f1")):
                group_note = " (provisional grouping)" if bool(r.get("provisional_grouping", False)) else ""
                eval_level = r.get("evaluation_level", "sample")
                lines.append(
                    f"- `{r['task']}` / `{r['model']}`: macro F1 = {r['metric_macro_f1']:.3f}"
                    + (f", ROC AUC = {r['metric_roc_auc']:.3f}" if pd.notna(r.get("metric_roc_auc")) else "")
                    + f", evaluation_level = `{eval_level}`{group_note}"
                )
            elif pd.notna(r.get("metric_mae")):
                lines.append(f"- `{r['task']}` / `{r['model']}`: MAE = {r['metric_mae']:.4f}")

    lines.extend(["", "## Recommendations", ""])
    lines.extend([f"- {rec}" for rec in recommendations])

    if not severity_df.empty:
        top = (
            severity_df.groupby(["sample_id"], as_index=False)["severity_score_0_1"]
            .mean()
            .sort_values("severity_score_0_1", ascending=False)
            .head(5)
        )
        lines.extend(["", "## Highest Severity Samples (Derived)", ""])
        for _, r in top.iterrows():
            lines.append(f"- sample `{r['sample_id']}`: severity {r['severity_score_0_1']:.3f}")

    lines.extend(["", "## Notes", ""])
    lines.extend([f"- {n}" for n in notes])

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate outputs and generate baseline markdown report.")
    parser.add_argument("--metrics-csv", type=Path, default=Path("artifacts/results/cv_metrics.csv"))
    parser.add_argument("--predictions-csv", type=Path, default=Path("artifacts/results/oof_predictions.csv"))
    parser.add_argument("--features-csv", type=Path, default=Path("artifacts/features/video_level_features.csv"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/baseline_report.md"))
    parser.add_argument(
        "--severity-csv", type=Path, default=Path("artifacts/results/severity_enriched_predictions.csv")
    )
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics_csv) if args.metrics_csv.exists() else pd.DataFrame()
    pred = pd.read_csv(args.predictions_csv) if args.predictions_csv.exists() else pd.DataFrame()
    feat = pd.read_csv(args.features_csv) if args.features_csv.exists() else pd.DataFrame()

    severity_df = derive_severity(pred, feat) if not pred.empty and not feat.empty else pd.DataFrame()
    if not severity_df.empty:
        args.severity_csv.parent.mkdir(parents=True, exist_ok=True)
        severity_df.to_csv(args.severity_csv, index=False)

    best_rows = []
    if not metrics.empty and "task" in metrics.columns:
        for task, sub in metrics.groupby("task"):
            s = sub.copy()
            s["score"] = s.apply(_score_column, axis=1)
            best_rows.append(s.sort_values("score", ascending=False).iloc[0])
    best_df = pd.DataFrame(best_rows).sort_values("task") if best_rows else pd.DataFrame()

    recommendations = []
    for _, r in best_df.iterrows():
        if pd.notna(r.get("metric_macro_f1")):
            recommendations.append(
                f"For `{r['task']}`, start with `{r['model']}` (macro F1={r['metric_macro_f1']:.3f})."
            )
        elif pd.notna(r.get("metric_mae")):
            recommendations.append(
                f"For `{r['task']}`, start with `{r['model']}` (MAE={r['metric_mae']:.4f})."
            )
    if not recommendations:
        recommendations.append("Insufficient labels for modeling; fill metadata template and retrain.")

    notes = [
        "All model metrics use grouped cross-validation to avoid window/video leakage.",
        "If window-level features were used, metrics are aggregated and reported at video level.",
        "Window count (e.g., 8077) is never treated as independent evaluation sample size.",
        "When person_id has <3 unique IDs, grouping falls back to video-level and is marked provisional.",
        "Severity score is derived from calibrated uncertainty and normalized feature deviations.",
        "Severity is not trained as an independent label target in this pipeline.",
    ]
    write_report(args.report_md, best_df, severity_df, notes, recommendations)
    print(f"Wrote report -> {args.report_md}")
    if not severity_df.empty:
        print(f"Wrote severity predictions -> {args.severity_csv}")


if __name__ == "__main__":
    main()
