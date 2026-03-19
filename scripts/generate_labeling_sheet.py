#!/usr/bin/env python3
"""Create manual labeling review sheet with analytics context features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _clean_text_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df))
    s = df[col].fillna("").astype(str).replace({"nan": "", "None": ""})
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata/labels_review_sheet.csv")
    parser.add_argument("--labels-csv", type=Path, default=Path("metadata/labels_template.csv"))
    parser.add_argument("--features-csv", type=Path, default=Path("artifacts/features/video_level_features.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("metadata/labels_review_sheet.csv"))
    args = parser.parse_args()

    if not args.labels_csv.exists():
        raise SystemExit(f"Missing labels template: {args.labels_csv}")
    if not args.features_csv.exists():
        raise SystemExit(f"Missing features CSV: {args.features_csv}")

    labels = pd.read_csv(args.labels_csv)
    feats = pd.read_csv(args.features_csv)

    if "sample_id" not in labels.columns:
        raise SystemExit("labels CSV must contain sample_id")
    if "sample_id" not in feats.columns:
        raise SystemExit("features CSV must contain sample_id")

    labels["sample_id"] = labels["sample_id"].astype(str)
    feats["sample_id"] = feats["sample_id"].astype(str)
    merged = labels.merge(feats, on="sample_id", how="left", suffixes=("", "_feat"))

    # Manual-label columns (never auto-guessed).
    food_type = _clean_text_col(merged, "food_type")
    chew_side_label = _clean_text_col(merged, "chew_side_label")
    if (chew_side_label == "").all() and "chewing_side_dominance" in merged.columns:
        chew_side_label = _clean_text_col(merged, "chewing_side_dominance")
    label_quality = _clean_text_col(merged, "label_quality")
    notes = _clean_text_col(merged, "notes")

    # Helpful context metrics for human review.
    chew_rate_hz = merged.get("chew_rate_hz_from_bites_mean", merged.get("estimated_chew_rate_hz_from_timeseries"))
    asymmetry_ratio = merged.get("left_right_asymmetry_ratio_from_jaw")
    pause_ratio = merged.get("pause_ratio", merged.get("pause_ratio_from_bites"))
    rhythm_variance = merged.get("rhythm_variance_from_smooth")
    jaw_opening_mean = merged.get("jaw_opening_amplitude_mean")
    dominant_frequency = merged.get("dominant_mouth_open_frequency_hz")

    # Optional candidate columns, clearly separated from human labels.
    left = pd.to_numeric(merged.get("chew_side_left_ratio"), errors="coerce")
    right = pd.to_numeric(merged.get("chew_side_right_ratio"), errors="coerce")
    candidate_conf = (left - right).abs()
    candidate = np.where(left > right, "left", np.where(right > left, "right", ""))
    candidate = np.where(pd.isna(candidate_conf), "", candidate)

    out = pd.DataFrame(
        {
            "sample_id": merged["sample_id"],
            "video_id": _clean_text_col(merged, "video_id"),
            "person_id": _clean_text_col(merged, "person_id"),
            "food_type": food_type,
            "chew_side_label": chew_side_label,
            "label_quality": label_quality,
            "notes": notes,
            "chew_rate_hz": pd.to_numeric(chew_rate_hz, errors="coerce"),
            "asymmetry_ratio": pd.to_numeric(asymmetry_ratio, errors="coerce"),
            "pause_ratio": pd.to_numeric(pause_ratio, errors="coerce"),
            "rhythm_variance": pd.to_numeric(rhythm_variance, errors="coerce"),
            "jaw_opening_mean": pd.to_numeric(jaw_opening_mean, errors="coerce"),
            "dominant_frequency": pd.to_numeric(dominant_frequency, errors="coerce"),
            "chew_side_candidate": candidate,
            "chew_side_candidate_confidence": pd.to_numeric(candidate_conf, errors="coerce"),
        }
    )

    out["video_id"] = np.where(out["video_id"].astype(str).str.strip() == "", out["sample_id"], out["video_id"])
    out["person_id"] = np.where(
        out["person_id"].astype(str).str.strip() == "", out["sample_id"], out["person_id"]
    )
    out = out.sort_values("sample_id", key=lambda s: s.astype(str)).reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
