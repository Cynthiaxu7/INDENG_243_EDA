#!/usr/bin/env python3
"""QC checks for missing source tables and derived feature completeness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


TARGET_SAMPLES = ["2", "3", "6", "20"]


def resolve_sample_dir(sample_root: Path) -> Optional[Path]:
    csvs = list(sample_root.glob("*.csv"))
    if csvs:
        return sample_root
    nested = [p for p in sample_root.iterdir() if p.is_dir()]
    if len(nested) == 1 and list(nested[0].glob("*.csv")):
        return nested[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feature QC report for key samples.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--features-csv", type=Path, default=Path("artifacts/features/video_level_features.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/feature_qc_report.md"))
    args = parser.parse_args()

    if not args.features_csv.exists():
        raise SystemExit(f"Missing features CSV: {args.features_csv}")
    feat = pd.read_csv(args.features_csv)
    feat["sample_id"] = feat["sample_id"].astype(str)

    core_cols = [
        "chew_rate_hz_from_bites_mean",
        "chew_cycle_count",
        "pause_ratio",
        "rhythm_variance_from_smooth",
        "left_right_asymmetry_ratio_from_jaw",
        "jaw_opening_amplitude_mean",
        "dominant_mouth_open_frequency_hz",
    ]

    lines = ["# Feature QC Report", "", "## Missing face_landmarks.csv target samples", ""]
    for sid in TARGET_SAMPLES:
        sample_root = args.data_dir / sid
        resolved = resolve_sample_dir(sample_root)
        if resolved is None:
            lines.append(f"- sample `{sid}`: directory unresolved")
            continue

        face_exists = (resolved / "face_landmarks.csv").exists()
        available = [p.name for p in resolved.glob("*.csv")]
        row = feat.loc[feat["sample_id"] == sid]
        if row.empty:
            lines.append(f"- sample `{sid}`: no derived feature row found")
            continue
        row = row.iloc[0]
        missing_feature_cols = [c for c in core_cols if c not in row.index or pd.isna(row[c])]

        if face_exists:
            status = "face_landmarks present"
        elif not missing_feature_cols:
            status = "computed from alternate tables (mouth_metrics/mouth_timeseries/chewing_analysis/lip_landmarks)"
        else:
            status = "partially missing derived features"

        lines.append(f"- sample `{sid}`")
        lines.append(f"  - face_landmarks.csv: {'present' if face_exists else 'missing'}")
        lines.append(f"  - available_tables: {available}")
        lines.append(f"  - derived_feature_status: {status}")
        if missing_feature_cols:
            lines.append(f"  - missing_derived_features: {missing_feature_cols}")
        else:
            lines.append("  - missing_derived_features: []")
        lines.append("  - imputation_note: no imputation in feature extraction; model pipelines may impute at train-time.")
        lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
