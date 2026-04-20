#!/usr/bin/env python3
"""Generate interpretable per-video analytics and categorical bands."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def chew_rate_band(x: float) -> str:
    if pd.isna(x):
        return "unknown"
    if x < 3.0:
        return "slow"
    if x <= 7.0:
        return "normal"
    return "fast"


def asymmetry_band(balance_abs_diff: float) -> str:
    if pd.isna(balance_abs_diff):
        return "unknown"
    if balance_abs_diff < 0.15:
        return "balanced"
    if balance_abs_diff < 0.40:
        return "mild bias"
    return "strong bias"


def pause_band(x: float) -> str:
    if pd.isna(x):
        return "unknown"
    if x < 0.25:
        return "continuous"
    if x < 0.50:
        return "moderate pauses"
    return "pause-heavy"


def rhythm_band(x: float, q33: float, q66: float) -> str:
    if pd.isna(x):
        return "unknown"
    if x <= q33:
        return "steady-like"
    if x <= q66:
        return "moderate variability"
    return "irregular-like"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create direct analytics report from video-level features.")
    parser.add_argument("--features-csv", type=Path, default=Path("artifacts/features/video_level_features.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("artifacts/results/per_video_analytics.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/direct_analytics_report.md"))
    args = parser.parse_args()

    if not args.features_csv.exists():
        raise SystemExit(f"Missing features CSV: {args.features_csv}")
    df = pd.read_csv(args.features_csv)

    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "video_id": df["video_id"].astype(str),
            "person_id": df["person_id"].astype(str).replace({"nan": ""}),
            "chew_rate_hz": pd.to_numeric(df["chew_rate_hz_from_bites_mean"], errors="coerce"),
            "chew_count": pd.to_numeric(df["chew_cycle_count"], errors="coerce"),
            "pause_ratio": pd.to_numeric(df["pause_ratio"], errors="coerce"),
            "rhythm_variance": pd.to_numeric(df["rhythm_variance_from_smooth"], errors="coerce"),
            "asymmetry_ratio": pd.to_numeric(df["left_right_asymmetry_ratio_from_jaw"], errors="coerce"),
            "asymmetry_balance_abs_diff": pd.to_numeric(df["chew_side_balance_abs_diff"], errors="coerce"),
            "jaw_opening_mean": pd.to_numeric(df["jaw_opening_amplitude_mean"], errors="coerce"),
            "jaw_opening_std": pd.to_numeric(df["jaw_opening_amplitude_std"], errors="coerce"),
            "jaw_opening_p95": pd.to_numeric(df["jaw_opening_amplitude_p95"], errors="coerce"),
            "dominant_frequency": pd.to_numeric(df["dominant_mouth_open_frequency_hz"], errors="coerce"),
        }
    )

    q33 = out["rhythm_variance"].quantile(0.33)
    q66 = out["rhythm_variance"].quantile(0.66)
    out["chew_rate_band"] = out["chew_rate_hz"].apply(chew_rate_band)
    out["asymmetry_band"] = out["asymmetry_balance_abs_diff"].apply(asymmetry_band)
    out["pause_style_band"] = out["pause_ratio"].apply(pause_band)
    out["rhythm_band"] = out["rhythm_variance"].apply(lambda x: rhythm_band(x, q33, q66))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    lines = [
        "# Direct Analytics Report",
        "",
        "- This report uses threshold-based analytics outputs, not supervised targets.",
        "- Severity should be derived later from uncertainty + normalized deviations.",
        "",
        "## Dataset Summary",
        "",
        f"- Number of videos: **{out['video_id'].nunique()}**",
        f"- Median chew_rate_hz: **{out['chew_rate_hz'].median():.3f}**",
        f"- Median pause_ratio: **{out['pause_ratio'].median():.3f}**",
        "",
        "## Categorical Band Counts",
        "",
    ]
    for col in ["chew_rate_band", "asymmetry_band", "pause_style_band", "rhythm_band"]:
        vc = out[col].value_counts(dropna=False).to_dict()
        lines.append(f"- `{col}`: {vc}")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Per-video analytics table: `{args.out_csv}`",
        ]
    )
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
