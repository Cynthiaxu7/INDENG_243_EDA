#!/usr/bin/env python3
"""Build leakage-safe windowed samples while preserving group IDs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def resolve_sample_dir(sample_root: Path) -> Optional[Path]:
    csvs_here = list(sample_root.glob("*.csv"))
    if csvs_here:
        return sample_root
    nested = [p for p in sample_root.iterdir() if p.is_dir()]
    if len(nested) == 1 and list(nested[0].glob("*.csv")):
        return nested[0]
    return None


def summarize_window(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in ["mouth_open_px", "mouth_width_px", "jaw_left_y_px", "jaw_right_y_px", "mouth_open_smooth"]:
        if col in df.columns:
            s = df[col].astype(float)
            out[f"{col}_mean"] = float(s.mean())
            out[f"{col}_std"] = float(s.std(ddof=0))
            out[f"{col}_p95"] = float(s.quantile(0.95))
    if {"jaw_left_y_px", "jaw_right_y_px"}.issubset(df.columns):
        d = (df["jaw_left_y_px"] - df["jaw_right_y_px"]).astype(float)
        out["jaw_lr_delta_abs_mean"] = float(d.abs().mean())
    if "mouth_is_open" in df.columns:
        out["pause_ratio"] = float((df["mouth_is_open"].astype(float) == 0).mean())
    if "time_sec" in df.columns and len(df) > 1:
        t = df["time_sec"].astype(float)
        out["window_duration_sec"] = float(t.max() - t.min())
    else:
        out["window_duration_sec"] = np.nan
    out["window_frame_count"] = int(len(df))
    return out


def fixed_windows(mt: pd.DataFrame, sample_id: str, person_id: str, seconds: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if mt.empty or "time_sec" not in mt.columns:
        return rows
    mt = mt.sort_values("time_sec").reset_index(drop=True)
    t0, t1 = float(mt["time_sec"].min()), float(mt["time_sec"].max())
    w = 0
    cur = t0
    while cur < t1:
        end = cur + seconds
        seg = mt[(mt["time_sec"] >= cur) & (mt["time_sec"] < end)]
        if len(seg) >= 20:
            rec = summarize_window(seg)
            rec.update(
                {
                    "sample_id": sample_id,
                    "video_id": sample_id,
                    "person_id": person_id or sample_id,
                    "window_id": f"{sample_id}_fixed_{w:04d}",
                    "window_type": "fixed",
                }
            )
            rows.append(rec)
            w += 1
        cur = end
    return rows


def bite_windows(
    mt: pd.DataFrame, ca: pd.DataFrame, sample_id: str, person_id: str, margin_sec: float
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if mt.empty or ca.empty or "time_sec" not in mt.columns:
        return rows
    mt = mt.sort_values("time_sec").reset_index(drop=True)
    for _, bite in ca.iterrows():
        start = float(bite["start_time_sec"]) - margin_sec
        end = float(bite["end_time_sec"]) + margin_sec
        seg = mt[(mt["time_sec"] >= start) & (mt["time_sec"] <= end)]
        if len(seg) < 20:
            continue
        rec = summarize_window(seg)
        rec.update(
            {
                "sample_id": sample_id,
                "video_id": sample_id,
                "person_id": person_id or sample_id,
                "window_id": f"{sample_id}_bite_{int(bite['bite_id']):04d}",
                "window_type": "bite_centered",
                "bite_id": int(bite["bite_id"]) if "bite_id" in bite else np.nan,
            }
        )
        rows.append(rec)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build window-level feature table from mouth timeseries.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--labels-csv", type=Path, default=Path("metadata/labels_template.csv"))
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--bite-margin-seconds", type=float, default=0.25)
    parser.add_argument("--include-bite-centered", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/features/window_level_features.csv"))
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    label_df = pd.read_csv(args.labels_csv) if args.labels_csv.exists() else pd.DataFrame()
    person_map = {}
    if not label_df.empty and {"sample_id", "person_id"}.issubset(label_df.columns):
        person_map = {str(r["sample_id"]): str(r["person_id"]) for _, r in label_df.iterrows()}

    all_rows: List[Dict[str, object]] = []
    for d in sorted([p for p in args.data_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        sample_dir = resolve_sample_dir(d)
        if sample_dir is None:
            continue
        sid = d.name
        pid = person_map.get(sid, sid)
        mt_path = sample_dir / "mouth_timeseries.csv"
        ca_path = sample_dir / "chewing_analysis.csv"
        if not mt_path.exists():
            continue
        mt = pd.read_csv(mt_path)
        ca = pd.read_csv(ca_path) if ca_path.exists() else pd.DataFrame()

        all_rows.extend(fixed_windows(mt, sid, pid, args.window_seconds))
        if args.include_bite_centered and not ca.empty:
            all_rows.extend(bite_windows(mt, ca, sid, pid, args.bite_margin_seconds))

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out_df)} windows to {args.output_csv}")


if __name__ == "__main__":
    main()
