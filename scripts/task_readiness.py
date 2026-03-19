#!/usr/bin/env python3
"""Assess which tasks are realistically trainable from current labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _clean_label_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})
    return s


def _task_series(df: pd.DataFrame, task: str) -> pd.Series:
    if task in df.columns:
        return _clean_label_series(df[task])
    # Alias support for previous column naming.
    if task == "chew_side_label" and "chewing_side_dominance" in df.columns:
        return _clean_label_series(df["chewing_side_dominance"])
    return pd.Series([np.nan] * len(df))


def _multiclass_readiness(y: pd.Series, min_per_class: int = 4) -> Tuple[bool, Dict[str, int]]:
    counts = y.dropna().value_counts().to_dict()
    if not counts:
        return False, {}
    trainable = min(counts.values()) >= min_per_class and len(counts) >= 3
    return trainable, {str(k): int(v) for k, v in counts.items()}


def _binary_readiness(
    y: pd.Series, allowed_classes: Tuple[str, ...], min_per_class: int = 5
) -> Tuple[bool, Dict[str, int]]:
    s = y.dropna().str.lower()
    s = s[s.isin(allowed_classes)]
    counts = s.value_counts().to_dict()
    trainable = len(counts) == 2 and min(counts.values()) >= min_per_class
    return trainable, {str(k): int(v) for k, v in counts.items()}


def _regression_readiness(y: pd.Series) -> Tuple[bool, Dict[str, float]]:
    x = pd.to_numeric(y, errors="coerce").dropna()
    if x.empty:
        return False, {"n_labeled": 0}
    var = float(x.var(ddof=0))
    trainable = len(x) >= 15 and var > 1e-8
    return trainable, {"n_labeled": int(len(x)), "variance": var}


def build_readiness(labels_df: pd.DataFrame) -> dict:
    tasks = {}

    food = _task_series(labels_df, "food_type").str.lower()
    trainable, dist = _multiclass_readiness(food, min_per_class=4)
    tasks["food_type"] = {
        "task_type": "multiclass",
        "n_labeled_samples": int(food.notna().sum()),
        "class_distribution": dist,
        "trainable_now": bool(trainable),
        "rule": "multiclass requires each class >= 4 samples",
    }

    side = _task_series(labels_df, "chew_side_label")
    trainable, dist = _binary_readiness(side, ("left", "right"), min_per_class=5)
    tasks["chew_side_label"] = {
        "task_type": "binary",
        "n_labeled_samples": int(side.notna().sum()),
        "class_distribution": dist,
        "trainable_now": bool(trainable),
        "rule": "binary requires each class >= 5 samples",
        "notes": "balanced can remain unlabeled for now",
    }

    rhythm = _task_series(labels_df, "rhythm_class")
    trainable, dist = _binary_readiness(rhythm, ("steady", "irregular"), min_per_class=5)
    tasks["rhythm_class"] = {
        "task_type": "binary",
        "n_labeled_samples": int(rhythm.notna().sum()),
        "class_distribution": dist,
        "trainable_now": bool(trainable),
        "rule": "binary requires each class >= 5 samples",
    }

    pause = _task_series(labels_df, "pause_style")
    trainable, dist = _binary_readiness(pause, ("continuous", "pause-heavy"), min_per_class=5)
    tasks["pause_style"] = {
        "task_type": "binary",
        "n_labeled_samples": int(pause.notna().sum()),
        "class_distribution": dist,
        "trainable_now": bool(trainable),
        "rule": "binary requires each class >= 5 samples",
    }

    chew_rate = _task_series(labels_df, "chew_rate_target")
    trainable, stats = _regression_readiness(chew_rate)
    tasks["chew_rate_target"] = {
        "task_type": "regression",
        "n_labeled_samples": int(stats.get("n_labeled", 0)),
        "variance": float(stats.get("variance", np.nan)),
        "trainable_now": bool(trainable),
        "rule": "regression requires >= 15 labels with non-trivial variance",
        "notes": "treat as direct analytics unless external ground truth exists",
    }

    return {
        "n_total_samples": int(len(labels_df)),
        "readiness_rules": {
            "multiclass": "each class >= 4 samples",
            "binary": "each class >= 5 samples",
            "regression": ">= 15 labels and non-trivial variance",
        },
        "tasks": tasks,
    }


def write_markdown(report: dict, out_path: Path) -> None:
    lines = ["# Task Readiness Report", ""]
    lines.append(f"- Total samples in label sheet: **{report['n_total_samples']}**")
    lines.append("")
    lines.append("## Readiness by Task")
    lines.append("")
    for task, info in report["tasks"].items():
        lines.append(f"### {task}")
        lines.append(f"- task_type: `{info['task_type']}`")
        lines.append(f"- labeled_samples: **{info['n_labeled_samples']}**")
        if "class_distribution" in info:
            lines.append(f"- class_distribution: `{info['class_distribution']}`")
        if "variance" in info:
            var = info["variance"]
            lines.append(f"- variance: `{var:.8f}`" if pd.notna(var) else "- variance: `nan`")
        lines.append(f"- trainable_now: **{info['trainable_now']}**")
        lines.append(f"- rule: {info['rule']}")
        if "notes" in info:
            lines.append(f"- notes: {info['notes']}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create task readiness report from label sheet.")
    parser.add_argument("--labels-csv", type=Path, default=Path("metadata/labels_template.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/task_readiness_report.md"))
    parser.add_argument("--out-json", type=Path, default=Path("reports/task_readiness_report.json"))
    args = parser.parse_args()

    if not args.labels_csv.exists():
        raise SystemExit(f"Missing labels CSV: {args.labels_csv}")
    labels = pd.read_csv(args.labels_csv)
    report = build_readiness(labels)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
