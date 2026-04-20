#!/usr/bin/env python3
"""Generate final recommendation report for next project phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Write next-phase recommendation markdown.")
    parser.add_argument("--readiness-json", type=Path, default=Path("reports/task_readiness_report.json"))
    parser.add_argument("--analytics-csv", type=Path, default=Path("artifacts/results/per_video_analytics.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/next_phase_recommendation.md"))
    args = parser.parse_args()

    readiness = {}
    if args.readiness_json.exists():
        readiness = json.loads(args.readiness_json.read_text(encoding="utf-8"))
    tasks = readiness.get("tasks", {})
    trainable = [t for t, v in tasks.items() if v.get("trainable_now")]
    not_trainable = [t for t, v in tasks.items() if not v.get("trainable_now")]

    n_videos = 20
    if args.analytics_csv.exists():
        analytics = pd.read_csv(args.analytics_csv)
        if "video_id" in analytics.columns:
            n_videos = int(analytics["video_id"].nunique())

    lines = [
        "# Next Phase Recommendation",
        "",
        "## What Can Be Trained Now",
        "",
    ]
    if trainable:
        for t in trainable:
            lines.append(f"- `{t}` is ready for grouped CV training.")
    else:
        lines.append("- No supervised task currently passes readiness thresholds.")
        lines.append("- First realistic training run should start with `food_type` once labels are collected.")
        lines.append("- `chew_side_label` can follow as soon as left/right labels are sufficiently populated.")

    lines.extend(
        [
            "",
            "## What Should Stay Rule-Based For Now",
            "",
            "- Keep chew rate, pause ratio, rhythm variance, asymmetry ratio, jaw opening stats, and dominant frequency as direct analytics outputs.",
            "- Do not train severity as an independent target; derive it from calibrated model uncertainty plus normalized feature deviations.",
            "- Treat window-derived rows as segmentation artifacts, not true independent sample size.",
            "",
            "## Label Collection Priorities",
            "",
            "- Priority 1: `food_type` with balanced coverage across soft/medium/hard.",
            "- Priority 2: `chew_side_label` with clear left/right labels (balanced can remain unlabeled).",
            "- Add `person_id` consistently to enable person-level grouped validation.",
            "- Use `label_quality` and `notes` fields to document uncertain labels.",
            "",
            "## Window-Level Modeling Guidance",
            "",
            f"- True sample scale is `{n_videos}` original videos; windows (e.g. 8077) only increase within-video observations.",
            "- Window-level training can help feature stabilization but must keep grouped splits and video-level reporting.",
            "- If person_id count is below 3 unique IDs, treat grouped results as provisional.",
            "",
            "## Tasks Not Trainable Yet",
            "",
        ]
    )
    if not_trainable:
        for t in not_trainable:
            lines.append(f"- `{t}`: insufficient labeled coverage.")
    else:
        lines.append("- All tracked tasks are currently trainable.")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
