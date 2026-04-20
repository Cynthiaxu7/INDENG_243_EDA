#!/usr/bin/env python3
"""Run the minimal complete analytics-first pipeline end-to-end.

This runner intentionally does not require food_type supervision.
It reuses existing scripts and produces a usable project report bundle.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(step_name: str, cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] {step_name}: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(f"[FAIL] {step_name} (exit={completed.returncode})")
    print(f"[OK] {step_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete non-supervised-first model workflow.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--include-windowing", action="store_true")
    parser.add_argument("--include-label-progress", action="store_true")
    parser.add_argument(
        "--summary-md", type=Path, default=Path("reports/model_complete_status.md")
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    py = sys.executable

    run_step("data_audit", [py, "scripts/data_audit.py"], root)
    run_step("extract_features", [py, "scripts/extract_features.py"], root)
    if args.include_windowing:
        run_step(
            "build_windows",
            [py, "scripts/build_windows.py", "--include-bite-centered"],
            root,
        )
    run_step("direct_analytics_report", [py, "scripts/direct_analytics_report.py"], root)
    run_step("feature_qc", [py, "scripts/feature_qc.py"], root)
    run_step("task_readiness", [py, "scripts/task_readiness.py"], root)

    if args.include_label_progress:
        run_step("label_progress_tracker", [py, "scripts/label_progress_tracker.py"], root)

    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(
        "\n".join(
            [
                "# Model Complete Status",
                "",
                "- Mode: analytics-first (food_type classification optional)",
                "- This run does not require supervised labels to be considered complete/usable.",
                "",
                "## Generated Artifacts",
                "",
                "- `reports/data_audit_report.md`",
                "- `artifacts/features/video_level_features.csv`",
                "- `reports/direct_analytics_report.md`",
                "- `reports/feature_qc_report.md`",
                "- `reports/task_readiness_report.md`",
                "- Optional label progress reports if enabled.",
                "",
                "## Notes",
                "",
                "- If food_type labels are insufficient, supervised training is intentionally skipped.",
                "- The pipeline remains usable for analytics/QC/reporting and labeling operations.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[OK] summary report -> {args.summary_md}")


if __name__ == "__main__":
    main()
