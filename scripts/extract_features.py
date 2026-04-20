#!/usr/bin/env python3
"""Feature extraction for chewing analysis in small-data settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


EXPECTED_TABLES = ["mouth_metrics.csv", "mouth_timeseries.csv", "chewing_analysis.csv", "lip_landmarks.csv"]


def resolve_sample_dir(sample_root: Path) -> Optional[Path]:
    """Handle both direct and one-level nested sample layouts."""
    csvs_here = list(sample_root.glob("*.csv"))
    if csvs_here:
        return sample_root
    nested = [p for p in sample_root.iterdir() if p.is_dir()]
    if len(nested) == 1 and list(nested[0].glob("*.csv")):
        return nested[0]
    return None


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def dominant_freq(signal: np.ndarray, dt: float) -> float:
    if signal.size < 4 or dt <= 0:
        return np.nan
    x = signal - np.nanmean(signal)
    if np.allclose(x, 0):
        return 0.0
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=dt)
    power = np.abs(fft) ** 2
    if len(power) <= 1:
        return np.nan
    idx = np.argmax(power[1:]) + 1
    return float(freqs[idx])


def estimate_rate_from_times(times: pd.Series) -> float:
    valid = times.dropna().sort_values().to_numpy()
    if valid.size < 2:
        return np.nan
    duration = valid[-1] - valid[0]
    if duration <= 0:
        return np.nan
    return float((valid.size - 1) / duration)


def extract_one_sample(sample_id: str, sample_dir: Path, person_id: str = "") -> Dict[str, object]:
    mm = safe_read_csv(sample_dir / "mouth_metrics.csv")
    mt = safe_read_csv(sample_dir / "mouth_timeseries.csv")
    ca = safe_read_csv(sample_dir / "chewing_analysis.csv")
    ll = safe_read_csv(sample_dir / "lip_landmarks.csv")

    row: Dict[str, object] = {
        "sample_id": sample_id,
        "video_id": sample_id,
        "person_id": person_id or sample_id,
        "data_path": str(sample_dir),
    }

    if mm is not None and not mm.empty:
        mouth_open = mm["mouth_open_px"].astype(float)
        jaw_delta = (mm["jaw_left_y_px"] - mm["jaw_right_y_px"]).astype(float)
        row.update(
            {
                "duration_sec": float(mm["time_sec"].max() - mm["time_sec"].min()),
                "frame_count": int(len(mm)),
                "jaw_opening_amplitude_mean": float(mouth_open.mean()),
                "jaw_opening_amplitude_std": float(mouth_open.std(ddof=0)),
                "jaw_opening_amplitude_p95": float(mouth_open.quantile(0.95)),
                "mouth_width_mean": float(mm["mouth_width_px"].astype(float).mean()),
                "mouth_open_to_width_ratio_mean": float(
                    (mouth_open / mm["mouth_width_px"].replace(0, np.nan)).mean()
                ),
                "lateral_jaw_displacement_mean_abs": float(jaw_delta.abs().mean()),
                "lateral_jaw_displacement_std": float(jaw_delta.std(ddof=0)),
                "left_right_asymmetry_ratio_from_jaw": float(
                    (mm["jaw_left_y_px"].mean() + 1e-6) / (mm["jaw_right_y_px"].mean() + 1e-6)
                ),
            }
        )

        dt = float(mm["time_sec"].diff().median()) if "time_sec" in mm else np.nan
        row["dominant_mouth_open_frequency_hz"] = dominant_freq(mouth_open.to_numpy(), dt=dt)

    if mt is not None and not mt.empty:
        mouth_open_flag = mt["mouth_is_open"].astype(float) if "mouth_is_open" in mt else pd.Series(dtype=float)
        row["pause_ratio"] = float((mouth_open_flag == 0).mean()) if not mouth_open_flag.empty else np.nan
        row["rhythm_variance_from_smooth"] = float(mt["mouth_open_smooth"].astype(float).var(ddof=0))
        row["estimated_chew_rate_hz_from_timeseries"] = estimate_rate_from_times(
            mt.loc[mt["phase_open"] > 0, "time_sec"] if "phase_open" in mt else mt["time_sec"]
        )

    if ca is not None and not ca.empty:
        freq = ca["chewing_frequency_per_sec"].astype(float)
        side = ca["chew_side"].astype(str).str.lower()
        left_count = int((side == "left").sum())
        right_count = int((side == "right").sum())
        total_side = max(left_count + right_count, 1)

        intervals = ca["start_time_sec"].astype(float).sort_values().diff().dropna()
        interval_mean = float(intervals.mean()) if not intervals.empty else np.nan
        interval_std = float(intervals.std(ddof=0)) if not intervals.empty else np.nan

        row.update(
            {
                "bite_count": int(len(ca)),
                "chew_cycle_count": int(ca["n_chews"].fillna(0).sum()),
                "chew_rate_hz_from_bites_mean": float(freq.mean()),
                "chew_rate_hz_from_bites_std": float(freq.std(ddof=0)),
                "chew_interval_mean_sec": interval_mean,
                "chew_interval_std_sec": interval_std,
                "chew_interval_cv": float(interval_std / interval_mean)
                if interval_mean and not np.isnan(interval_mean)
                else np.nan,
                "pause_ratio_from_bites": float(
                    max((ca["duration_sec"].sum() - ca["avg_chew_time_sec"].sum()), 0.0)
                    / max(ca["duration_sec"].sum(), 1e-6)
                ),
                "chew_side_left_ratio": left_count / total_side,
                "chew_side_right_ratio": right_count / total_side,
                "chew_side_balance_abs_diff": abs(left_count - right_count) / total_side,
            }
        )

    if ll is not None and not ll.empty:
        row.update(
            {
                "lip_landmark_rows": int(len(ll)),
                "lip_x_range_px": float(ll["x_px"].max() - ll["x_px"].min()),
                "lip_y_range_px": float(ll["y_px"].max() - ll["y_px"].min()),
                "lip_z_std_norm": float(ll["z_norm"].astype(float).std(ddof=0)),
            }
        )

    # A stable fallback chew rate target candidate for regression labeling.
    row["chew_rate_target_candidate_hz"] = row.get(
        "chew_rate_hz_from_bites_mean", row.get("estimated_chew_rate_hz_from_timeseries", np.nan)
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract handcrafted features from tabular chewing data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--labels-csv", type=Path, default=Path("metadata/labels_template.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/features/video_level_features.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/features/feature_manifest.json"))
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    label_df = pd.read_csv(args.labels_csv) if args.labels_csv.exists() else pd.DataFrame()
    person_map = {}
    if not label_df.empty and {"sample_id", "person_id"}.issubset(label_df.columns):
        person_map = {str(r["sample_id"]): str(r["person_id"]) for _, r in label_df.iterrows()}

    rows = []
    for d in sorted([p for p in args.data_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        resolved = resolve_sample_dir(d)
        if resolved is None:
            continue
        sid = d.name
        rows.append(extract_one_sample(sid, resolved, person_id=person_map.get(sid, sid)))

    feat_df = pd.DataFrame(rows).sort_values("sample_id", key=lambda s: s.astype(str))
    feat_df.to_csv(args.output_csv, index=False)

    manifest = {
        "n_samples": int(len(feat_df)),
        "n_features_excluding_ids": int(
            len([c for c in feat_df.columns if c not in {"sample_id", "video_id", "person_id", "data_path"}])
        ),
        "id_schema": ["sample_id", "video_id", "person_id"],
        "source_expected_tables": EXPECTED_TABLES,
        "notes": [
            "Inputs appear tabular, so video-landmark extraction is skipped by design.",
            "All outputs are per-original-video aggregates to preserve small-data assumptions.",
        ],
    }
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Extracted {len(feat_df)} samples -> {args.output_csv}")
    print(f"Wrote feature manifest -> {args.output_json}")


if __name__ == "__main__":
    main()
