#!/usr/bin/env python3
"""Audit raw inputs without modifying files in ./data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

EXPECTED_CSVS = {
    "chewing_analysis.csv",
    "face_landmarks.csv",
    "lip_landmarks.csv",
    "mouth_metrics.csv",
    "mouth_timeseries.csv",
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def resolve_sample_dirs(data_dir: Path) -> Dict[str, Path]:
    """Map sample_id -> directory containing sample CSVs."""
    mapping: Dict[str, Path] = {}
    for d in sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        csvs_here = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
        if csvs_here:
            mapping[d.name] = d
            continue

        nested = [p for p in d.iterdir() if p.is_dir()]
        if len(nested) == 1:
            nested_csvs = [
                p for p in nested[0].iterdir() if p.is_file() and p.suffix.lower() == ".csv"
            ]
            if nested_csvs:
                mapping[d.name] = nested[0]
                continue
    return mapping


def audit_data(data_dir: Path) -> Tuple[dict, pd.DataFrame]:
    sample_dirs = resolve_sample_dirs(data_dir)
    top_level_files = [p for p in data_dir.iterdir() if p.is_file()]
    top_level_subdirs = [p for p in data_dir.iterdir() if p.is_dir()]

    ext_counter = Counter()
    modality_counter = Counter()
    rows = []
    missing_by_sample = defaultdict(list)
    schema = defaultdict(set)

    for sample_id, sample_dir in sorted(
        sample_dirs.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]
    ):
        sample_files = [p for p in sample_dir.iterdir() if p.is_file()]
        names = {p.name for p in sample_files}
        for p in sample_files:
            ext_counter[p.suffix.lower() if p.suffix else "<no_ext>"] += 1

        has_video = any(p.suffix.lower() in VIDEO_EXTS for p in sample_files)
        has_csv = any(p.suffix.lower() == ".csv" for p in sample_files)
        modality_counter[
            "video_only"
            if has_video and not has_csv
            else "tabular_only"
            if has_csv and not has_video
            else "mixed"
            if has_video and has_csv
            else "unknown"
        ] += 1

        for expected in sorted(EXPECTED_CSVS):
            if expected not in names:
                missing_by_sample[sample_id].append(expected)

        for csv_path in [p for p in sample_files if p.suffix.lower() == ".csv"]:
            try:
                header = pd.read_csv(csv_path, nrows=0).columns.tolist()
                schema[csv_path.name].update(header)
            except Exception:
                missing_by_sample[sample_id].append(f"unreadable:{csv_path.name}")

        rows.append(
            {
                "sample_id": sample_id,
                "video_id": sample_id,
                "person_id": "",
                "data_path": str(sample_dir.relative_to(data_dir.parent)),
                "n_files": len(sample_files),
                "missing_expected_files": "|".join(missing_by_sample[sample_id]),
            }
        )

    audit = {
        "data_dir": str(data_dir),
        "top_level_subfolder_count": len(top_level_subdirs),
        "top_level_file_count": len(top_level_files),
        "sample_count_detected": len(sample_dirs),
        "sample_ids": sorted(sample_dirs.keys(), key=lambda x: int(x) if x.isdigit() else x),
        "global_file_types": dict(ext_counter),
        "likely_input_modality": dict(modality_counter),
        "missing_metadata_assumptions": [
            "person_id is currently unknown and must be supplied manually.",
            "target labels are not present in raw data; fill metadata template.",
            "sampling rates may vary by sample; verify fps/time consistency.",
            "folder names are numeric sample IDs, not guaranteed person IDs.",
        ],
        "schema_by_file_type": {k: sorted(v) for k, v in sorted(schema.items())},
        "missing_expected_files_by_sample": {k: v for k, v in sorted(missing_by_sample.items()) if v},
        "notes": [
            "No raw videos detected in ./data sample folders.",
            "Face landmark files are missing for some samples; fallback features should work.",
        ],
    }

    template_df = pd.DataFrame(rows).sort_values("sample_id", key=lambda s: s.astype(str))
    return audit, template_df


def write_markdown_report(audit: dict, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Data Audit Report")
    lines.append("")
    lines.append(f"- Data folder: `{audit['data_dir']}`")
    lines.append(f"- Top-level subfolder count: **{audit['top_level_subfolder_count']}**")
    lines.append(f"- Top-level file count: **{audit['top_level_file_count']}**")
    lines.append(f"- Detected sample folders: **{audit['sample_count_detected']}**")
    lines.append("")
    lines.append("## File Types")
    for k, v in sorted(audit["global_file_types"].items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Likely Input Modality")
    for k, v in sorted(audit["likely_input_modality"].items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Missing Metadata Assumptions")
    for item in audit["missing_metadata_assumptions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Missing Expected Files by Sample")
    if audit["missing_expected_files_by_sample"]:
        for sid, missing in audit["missing_expected_files_by_sample"].items():
            lines.append(f"- sample `{sid}`: {', '.join(missing)}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Notes")
    for n in audit["notes"]:
        lines.append(f"- {n}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ./data and build label metadata template.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/data_audit_report.md"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/data_audit_report.json"))
    parser.add_argument(
        "--labels-template-csv", type=Path, default=Path("metadata/labels_template.csv")
    )
    args = parser.parse_args()

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.labels_template_csv.parent.mkdir(parents=True, exist_ok=True)

    audit, template_df = audit_data(args.data_dir)

    # Extend with empty label columns required by downstream tasks.
    for col in [
        "food_type",
        "chewing_side_dominance",
        "rhythm_class",
        "pause_style",
        "chew_rate_target",
        "eater_speed_class",
        "notes",
    ]:
        if col not in template_df.columns:
            template_df[col] = ""

    write_markdown_report(audit, args.report_md)
    args.report_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    template_df.to_csv(args.labels_template_csv, index=False)

    print(json.dumps(audit, indent=2))
    print(f"\nWrote markdown report to: {args.report_md}")
    print(f"Wrote JSON report to: {args.report_json}")
    print(f"Wrote labels template to: {args.labels_template_csv}")


if __name__ == "__main__":
    main()
