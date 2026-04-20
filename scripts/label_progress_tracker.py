#!/usr/bin/env python3
"""Track labeling progress and provide sprint-ready labeling guidance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


TASK_SPECS = {
    "food_type": {"type": "multiclass", "classes": ["soft", "medium", "hard"], "min_per_class": 4},
    "chew_side_label": {"type": "binary", "classes": ["left", "right"], "min_per_class": 5},
    "rhythm_class": {"type": "binary", "classes": ["steady", "irregular"], "min_per_class": 5},
    "pause_style": {"type": "binary", "classes": ["continuous", "pause-heavy"], "min_per_class": 5},
    "chew_rate_target": {"type": "regression", "min_labeled": 15},
}

LOW_QUALITY_TOKENS = {"low", "poor", "uncertain", "review", "bad", "noisy"}


def clean_text(s: pd.Series) -> pd.Series:
    out = s.fillna("").astype(str).str.strip()
    return out.replace({"nan": "", "None": "", "null": ""})


def safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([""] * len(df), index=df.index)


def class_distribution(y: pd.Series, allowed_classes: List[str]) -> Dict[str, int]:
    s = clean_text(y).str.lower()
    s = s[s.isin(allowed_classes)]
    vc = s.value_counts().to_dict()
    return {c: int(vc.get(c, 0)) for c in allowed_classes}


def multiclass_detail(y: pd.Series, classes: List[str], target: int) -> Tuple[dict, bool]:
    dist = class_distribution(y, classes)
    deficits = {c: max(0, target - dist[c]) for c in classes}
    trainable = all(v >= target for v in dist.values())
    return {
        "labeled_samples": int(sum(dist.values())),
        "class_distribution": dist,
        "deficit_to_readiness_per_class": deficits,
        "min_additional_labels_to_unlock": int(sum(deficits.values())),
    }, trainable


def binary_detail(y: pd.Series, classes: List[str], target: int) -> Tuple[dict, bool]:
    dist = class_distribution(y, classes)
    deficits = {c: max(0, target - dist[c]) for c in classes}
    trainable = all(v >= target for v in dist.values())
    return {
        "labeled_samples": int(sum(dist.values())),
        "class_distribution": dist,
        "deficit_to_readiness_per_class": deficits,
        "min_additional_labels_to_unlock": int(sum(deficits.values())),
    }, trainable


def regression_detail(y: pd.Series, target_n: int) -> Tuple[dict, bool]:
    x = pd.to_numeric(y, errors="coerce").dropna()
    var = float(x.var(ddof=0)) if len(x) > 0 else np.nan
    trainable = len(x) >= target_n and pd.notna(var) and var > 1e-8
    return {
        "labeled_samples": int(len(x)),
        "variance": var,
        "deficit_to_readiness_count": int(max(0, target_n - len(x))),
    }, trainable


def person_id_check(df: pd.DataFrame, reference_labeled_count: int) -> dict:
    person = clean_text(safe_col(df, "person_id"))
    nonempty = person[person != ""]
    missing = int((person == "").sum())
    unique_nonempty = int(nonempty.nunique())
    can_use = unique_nonempty >= 3

    warning = ""
    interpretation_note = ""
    if not can_use:
        warning = "Fewer than 3 unique labeled people; grouped evaluation remains provisional (video-level fallback)."
    else:
        denominator = max(1, reference_labeled_count)
        ratio = unique_nonempty / denominator
        if ratio >= 0.8:
            interpretation_note = (
                "Person-level grouping is available, but repeated-person information is limited; "
                "treat results as preliminary cross-subject pilot findings."
            )
    return {
        "total_rows": int(len(df)),
        "missing_person_id_rows": missing,
        "unique_nonempty_person_ids": unique_nonempty,
        "can_use_person_grouping_now": can_use,
        "warning": warning,
        "interpretation_note": interpretation_note,
    }


def quality_checks(df: pd.DataFrame) -> dict:
    q = clean_text(safe_col(df, "label_quality"))
    dist = q.replace("", "<missing>").value_counts().to_dict()
    low_mask = q.str.lower().apply(lambda v: any(tok in v for tok in LOW_QUALITY_TOKENS) if v else False)
    missing_mask = q == ""
    review_mask = low_mask | missing_mask

    review_cols = ["sample_id", "video_id", "person_id", "food_type", "chew_side_label", "label_quality", "notes"]
    review_rows = df.loc[review_mask, [c for c in review_cols if c in df.columns]].copy()
    quality_ready = float(missing_mask.mean()) <= 0.4
    readiness_note = (
        "Label quality coverage is sparse. Fill label_quality during the same sprint: "
        "high = confident label, medium = mostly confident, low = ambiguous / poor-quality clip."
        if not quality_ready
        else "Label quality coverage is adequate for current sprint."
    )
    return {
        "label_quality_distribution": {str(k): int(v) for k, v in dist.items()},
        "low_quality_row_count": int(low_mask.sum()),
        "missing_quality_row_count": int(missing_mask.sum()),
        "review_needed_rows_count": int(review_mask.sum()),
        "review_needed_rows": review_rows.fillna("").astype(str).to_dict(orient="records"),
        "readiness_note": readiness_note,
    }


def task_blocker_and_action(task: str, trainable: bool, detail: dict) -> Tuple[str, str]:
    if trainable:
        return "none", "task trainable now"
    if task in {"food_type", "chew_side_label", "rhythm_class", "pause_style"}:
        deficits = detail.get("deficit_to_readiness_per_class", {})
        need = ", ".join([f"{k}+{v}" for k, v in deficits.items() if v > 0]) or "none"
        return "class coverage", f"label classes: {need}"
    if task == "chew_rate_target":
        return "insufficient numeric labels", f"add at least {detail.get('deficit_to_readiness_count', 0)} valid targets with variance"
    return "insufficient labels", "continue labeling"


def food_sprint_recommendation(
    counts: Dict[str, int],
    unlock_deficits: Dict[str, int],
    rec_deficits: Dict[str, int],
    unlock_target: int,
    recommended_target: int,
) -> str:
    if all(v <= 0 for v in unlock_deficits.values()):
        return "food_type is now trainable."
    # No arbitrary class tie-break: always provide balanced target instruction.
    return (
        f"Goal: reach {unlock_target} soft, {unlock_target} medium, {unlock_target} hard first. "
        f"Preferred: reach {recommended_target} per class for a more stable first run."
    )


def enrich_with_analytics(work: pd.DataFrame, analytics: pd.DataFrame | None) -> pd.DataFrame:
    out = work.copy()
    cols = [
        "sample_id",
        "chew_rate_hz",
        "asymmetry_ratio",
        "pause_ratio",
        "rhythm_variance",
        "jaw_opening_mean",
        "dominant_frequency",
        "chew_rate_band",
        "asymmetry_band",
        "pause_style_band",
        "rhythm_band",
    ]
    if analytics is None or analytics.empty:
        for c in cols:
            if c != "sample_id" and c not in out.columns:
                out[c] = np.nan
        return out
    a = analytics.copy()
    a["sample_id"] = a["sample_id"].astype(str)
    keep = [c for c in cols if c in a.columns]
    # Avoid duplicate-column suffix collisions: merge only missing columns.
    merge_cols = ["sample_id"] + [c for c in keep if c != "sample_id" and c not in out.columns]
    if len(merge_cols) == 1:
        # Already present in work table.
        return out
    return out.merge(a[merge_cols], on="sample_id", how="left")


def diverse_food_type_queue(labels: pd.DataFrame, analytics: pd.DataFrame | None, top_k: int = 12) -> pd.DataFrame:
    work = labels.copy()
    for c in ["sample_id", "video_id", "person_id", "food_type", "chew_side_label", "label_quality", "notes"]:
        work[c] = clean_text(safe_col(work, c))
    work = enrich_with_analytics(work, analytics)

    unlabeled_food = work["food_type"] == ""
    pool = work.loc[unlabeled_food].copy()
    if pool.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "video_id",
                "person_id",
                "chew_rate_hz",
                "asymmetry_ratio",
                "pause_ratio",
                "rhythm_variance",
                "jaw_opening_mean",
                "dominant_frequency",
                "priority_score",
                "priority_reason",
            ]
        )

    score = np.ones(len(pool), dtype=float) * 5.0
    reasons: List[List[str]] = [["unlabeled food_type"] for _ in range(len(pool))]

    # Missing person_id or chew_side_label still matters for downstream grouping/tasks.
    miss_person = pool["person_id"] == ""
    score += miss_person.astype(float) * 2.0
    for i in np.where(miss_person)[0]:
        reasons[i].append("missing person_id")

    miss_side = pool["chew_side_label"] == ""
    score += miss_side.astype(float) * 0.8
    for i in np.where(miss_side)[0]:
        reasons[i].append("missing chew_side_label")

    miss_quality = pool["label_quality"] == ""
    score += miss_quality.astype(float) * 1.0
    for i in np.where(miss_quality)[0]:
        reasons[i].append("missing label_quality")

    # Diversity boost using rarity of analytics bands.
    for band_col in ["chew_rate_band", "asymmetry_band", "pause_style_band", "rhythm_band"]:
        if band_col in pool.columns:
            freq = pool[band_col].fillna("unknown").value_counts()
            rarity = pool[band_col].fillna("unknown").map(lambda x: 1.0 / max(float(freq.get(x, 1)), 1.0))
            score += rarity.to_numpy() * 1.2
            for i in np.where(rarity.to_numpy() > rarity.median())[0]:
                reasons[i].append(f"diverse {band_col}")

    # Numeric spread cue: prioritize farther-from-median samples for coverage.
    numeric_cols = ["chew_rate_hz", "asymmetry_ratio", "pause_ratio", "rhythm_variance", "jaw_opening_mean", "dominant_frequency"]
    for col in numeric_cols:
        if col in pool.columns:
            x = pd.to_numeric(pool[col], errors="coerce")
            if x.notna().sum() >= 3:
                z = (x - x.median()).abs() / (x.std(ddof=0) + 1e-9)
                z = z.fillna(0.0)
                score += z.to_numpy() * 0.15
                for i in np.where(z.to_numpy() > z.quantile(0.75))[0]:
                    reasons[i].append(f"coverage spread in {col}")

    pool["priority_score"] = score
    pool["priority_reason"] = ["; ".join(sorted(set(r))) for r in reasons]
    pool = pool.sort_values(["priority_score", "sample_id"], ascending=[False, True]).head(top_k)
    cols = [
        "sample_id",
        "video_id",
        "person_id",
        "chew_rate_hz",
        "asymmetry_ratio",
        "pause_ratio",
        "rhythm_variance",
        "jaw_opening_mean",
        "dominant_frequency",
        "priority_score",
        "priority_reason",
    ]
    for c in cols:
        if c not in pool.columns:
            pool[c] = np.nan
    return pool[cols].reset_index(drop=True)


def build_general_next_rows(labels: pd.DataFrame, analytics: pd.DataFrame | None, top_k: int = 12) -> pd.DataFrame:
    work = labels.copy()
    for c in ["sample_id", "video_id", "person_id", "food_type", "chew_side_label", "label_quality", "notes"]:
        work[c] = clean_text(safe_col(work, c))
    work = enrich_with_analytics(work, analytics)
    score = np.zeros(len(work), dtype=float)
    reasons: List[List[str]] = [[] for _ in range(len(work))]

    miss_food = work["food_type"] == ""
    score += miss_food.astype(float) * 5.0
    for i in np.where(miss_food)[0]:
        reasons[i].append("missing food_type")

    miss_person = work["person_id"] == ""
    score += miss_person.astype(float) * 3.0
    for i in np.where(miss_person)[0]:
        reasons[i].append("missing person_id")

    miss_side = work["chew_side_label"] == ""
    score += miss_side.astype(float) * 2.0
    for i in np.where(miss_side)[0]:
        reasons[i].append("missing chew_side_label")

    q = work["label_quality"].str.lower()
    low_quality = q.apply(lambda v: any(tok in v for tok in LOW_QUALITY_TOKENS) if v else False)
    score += low_quality.astype(float) * 2.5
    for i in np.where(low_quality)[0]:
        reasons[i].append("low label_quality")

    missing_quality = work["label_quality"] == ""
    score += missing_quality.astype(float) * 1.5
    for i in np.where(missing_quality)[0]:
        reasons[i].append("missing label_quality")

    work["priority_score"] = score
    work["reasons"] = ["; ".join(sorted(set(r))) for r in reasons]
    out = work.sort_values(["priority_score", "sample_id"], ascending=[False, True]).head(top_k)
    out = out[out["priority_score"] > 0].copy()
    return out[["sample_id", "video_id", "person_id", "food_type", "chew_side_label", "label_quality", "priority_score", "reasons"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create label progress tracker reports.")
    parser.add_argument("--labels-csv", type=Path, default=Path("metadata/labels_review_sheet.csv"))
    parser.add_argument("--readiness-json", type=Path, default=Path("reports/task_readiness_report.json"))
    parser.add_argument("--analytics-csv", type=Path, default=Path("artifacts/results/per_video_analytics.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/label_progress_report.md"))
    parser.add_argument("--out-json", type=Path, default=Path("reports/label_progress_report.json"))
    parser.add_argument("--out-table-csv", type=Path, default=Path("artifacts/results/label_progress_table.csv"))
    parser.add_argument("--focus", choices=["food_type"], default=None)
    parser.add_argument("--unlock-target", type=int, default=4)
    parser.add_argument("--recommended-target", type=int, default=5)
    parser.add_argument("--food-queue-csv", type=Path, default=Path("artifacts/results/food_type_sprint_queue.csv"))
    parser.add_argument("--food-report-md", type=Path, default=Path("reports/food_type_sprint_report.md"))
    args = parser.parse_args()

    if not args.labels_csv.exists():
        raise SystemExit(f"Missing labels review sheet: {args.labels_csv}")

    labels = pd.read_csv(args.labels_csv)
    for c in ["sample_id", "video_id", "person_id", "food_type", "chew_side_label", "label_quality", "notes"]:
        if c not in labels.columns:
            labels[c] = ""
    labels["sample_id"] = labels["sample_id"].astype(str)
    labels["video_id"] = clean_text(labels["video_id"])
    labels["video_id"] = np.where(labels["video_id"] == "", labels["sample_id"], labels["video_id"])

    analytics = pd.read_csv(args.analytics_csv) if args.analytics_csv.exists() else pd.DataFrame()
    if not analytics.empty and "sample_id" in analytics.columns:
        analytics["sample_id"] = analytics["sample_id"].astype(str)

    previous_readiness = {}
    if args.readiness_json.exists():
        try:
            previous_readiness = json.loads(args.readiness_json.read_text(encoding="utf-8"))
        except Exception:
            previous_readiness = {}

    total = int(len(labels))
    task_rows: List[dict] = []
    task_json: Dict[str, dict] = {}

    for task, spec in TASK_SPECS.items():
        y = safe_col(labels, task)
        if spec["type"] == "multiclass":
            target = args.unlock_target if task == "food_type" else spec["min_per_class"]
            detail, trainable = multiclass_detail(y, spec["classes"], target=target)
        elif spec["type"] == "binary":
            detail, trainable = binary_detail(y, spec["classes"], target=spec["min_per_class"])
        else:
            detail, trainable = regression_detail(y, target_n=spec["min_labeled"])

        labeled_count = int(detail["labeled_samples"])
        blocker, action = task_blocker_and_action(task, trainable, detail)
        task_json[task] = {
            "task_type": spec["type"],
            "total_samples": total,
            "labeled_samples": labeled_count,
            "unlabeled_samples": int(total - labeled_count),
            "trainable_now": bool(trainable),
            **detail,
            "blocker": blocker,
            "next_action": action,
        }
        balance = str(detail.get("class_distribution")) if "class_distribution" in detail else f"var={detail.get('variance', np.nan):.6f}"
        task_rows.append(
            {
                "task": task,
                "labeled_count": labeled_count,
                "class_balance_or_variance": balance,
                "readiness": "trainable" if trainable else "blocked",
                "blocker": blocker,
                "next_action": action,
            }
        )

    # Food-type dual targets for sprint mode reporting.
    food_counts = class_distribution(labels["food_type"], ["soft", "medium", "hard"])
    food_unlock_def = {k: max(0, args.unlock_target - food_counts[k]) for k in food_counts}
    food_rec_def = {k: max(0, args.recommended_target - food_counts[k]) for k in food_counts}
    food_rec = food_sprint_recommendation(
        counts=food_counts,
        unlock_deficits=food_unlock_def,
        rec_deficits=food_rec_def,
        unlock_target=args.unlock_target,
        recommended_target=args.recommended_target,
    )

    labeled_for_person_ref = int(task_json.get("food_type", {}).get("labeled_samples", 0))
    person = person_id_check(labels, reference_labeled_count=max(labeled_for_person_ref, 1))
    quality = quality_checks(labels)
    suggestions = build_general_next_rows(labels, analytics, top_k=12)

    sprint_queue = pd.DataFrame()
    if args.focus == "food_type":
        sprint_queue = diverse_food_type_queue(labels, analytics, top_k=15)
        args.food_queue_csv.parent.mkdir(parents=True, exist_ok=True)
        sprint_queue.to_csv(args.food_queue_csv, index=False)
        sprint_lines = [
            "# Food Type Sprint Report",
            "",
            "- Suggested rows are for **diverse review coverage only**.",
            "- They are **not predicted class memberships**.",
            "",
            f"- Current counts (soft/medium/hard): `{food_counts}`",
            f"- Unlock target deficits ({args.unlock_target}/class): `{food_unlock_def}`",
            f"- Recommended target deficits ({args.recommended_target}/class): `{food_rec_def}`",
            f"- Minimum labels to unlock: **{sum(food_unlock_def.values())}**",
            f"- Recommendation: **{food_rec}**",
            "",
            "## Diverse Review Queue",
            "",
            "| sample_id | video_id | person_id | chew_rate_hz | asymmetry_ratio | pause_ratio | rhythm_variance | jaw_opening_mean | dominant_frequency | priority_score | priority_reason |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for _, r in sprint_queue.iterrows():
            sprint_lines.append(
                f"| {r['sample_id']} | {r['video_id']} | {r['person_id']} | "
                f"{float(r['chew_rate_hz']) if pd.notna(r['chew_rate_hz']) else np.nan:.4f} | "
                f"{float(r['asymmetry_ratio']) if pd.notna(r['asymmetry_ratio']) else np.nan:.4f} | "
                f"{float(r['pause_ratio']) if pd.notna(r['pause_ratio']) else np.nan:.4f} | "
                f"{float(r['rhythm_variance']) if pd.notna(r['rhythm_variance']) else np.nan:.4f} | "
                f"{float(r['jaw_opening_mean']) if pd.notna(r['jaw_opening_mean']) else np.nan:.4f} | "
                f"{float(r['dominant_frequency']) if pd.notna(r['dominant_frequency']) else np.nan:.4f} | "
                f"{float(r['priority_score']):.2f} | {r['priority_reason']} |"
            )
        args.food_report_md.parent.mkdir(parents=True, exist_ok=True)
        args.food_report_md.write_text("\n".join(sprint_lines), encoding="utf-8")

    report_json = {
        "total_samples": total,
        "focus": args.focus,
        "food_type_priority": {
            "class_counts": food_counts,
            "unlock_target": args.unlock_target,
            "recommended_target": args.recommended_target,
            "deficit_to_unlock_target_per_class": food_unlock_def,
            "deficit_to_recommended_target_per_class": food_rec_def,
            "min_additional_labels_to_unlock": int(sum(food_unlock_def.values())),
            "recommendation": food_rec,
        },
        "tasks": task_json,
        "person_id_completeness": person,
        "label_quality_checks": quality,
        "next_best_rows_to_label": suggestions.to_dict(orient="records"),
        "food_type_sprint_queue_rows": sprint_queue.to_dict(orient="records"),
        "previous_task_readiness_snapshot": previous_readiness.get("tasks", {}),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    args.out_table_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(task_rows).to_csv(args.out_table_csv, index=False)

    lines = [
        "# Label Progress Report",
        "",
        f"- Total samples: **{total}**",
        "",
        "## Food Type Priority",
        "",
        f"- soft/medium/hard counts: `{food_counts}`",
        f"- deficits to unlock target ({args.unlock_target}/class): `{food_unlock_def}`",
        f"- deficits to recommended target ({args.recommended_target}/class): `{food_rec_def}`",
        f"- minimum additional labels needed to unlock: **{sum(food_unlock_def.values())}**",
        f"- recommendation: **{food_rec}**",
        "",
        "## Task Status Table",
        "",
        "| task | labeled count | class balance / readiness | blocker | next action |",
        "|---|---:|---|---|---|",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task']} | {row['labeled_count']} | {row['class_balance_or_variance']} ({row['readiness']}) | {row['blocker']} | {row['next_action']} |"
        )

    lines.extend(
        [
            "",
            "## Person ID Completeness",
            "",
            f"- rows with missing person_id: **{person['missing_person_id_rows']}** / {person['total_rows']}",
            f"- unique non-empty person_id values: **{person['unique_nonempty_person_ids']}**",
            f"- can use person-level grouping now: **{person['can_use_person_grouping_now']}**",
        ]
    )
    if person["warning"]:
        lines.append(f"- warning: {person['warning']}")
    if person["interpretation_note"]:
        lines.append(f"- note: {person['interpretation_note']}")

    lines.extend(
        [
            "",
            "## Label Quality Checks",
            "",
            f"- label_quality distribution: `{quality['label_quality_distribution']}`",
            f"- rows with low quality labels: **{quality['low_quality_row_count']}**",
            f"- rows with missing label_quality: **{quality['missing_quality_row_count']}**",
            f"- review-needed rows: **{quality['review_needed_rows_count']}**",
            f"- readiness note: {quality['readiness_note']}",
            "",
            "## Next Best Rows To Label",
            "",
            "Top deterministic suggestions (no label guessing):",
            "",
            "| sample_id | video_id | person_id | food_type | chew_side_label | label_quality | priority_score | reasons |",
            "|---|---|---|---|---|---|---:|---|",
        ]
    )
    for _, r in suggestions.iterrows():
        lines.append(
            f"| {r['sample_id']} | {r['video_id']} | {r['person_id']} | {r['food_type']} | {r['chew_side_label']} | {r['label_quality']} | {r['priority_score']:.2f} | {r['reasons']} |"
        )

    if args.focus == "food_type":
        lines.extend(
            [
                "",
                "## Food Type Sprint Outputs",
                "",
                f"- Diverse queue CSV: `{args.food_queue_csv}`",
                f"- Sprint report: `{args.food_report_md}`",
                "- Queue rows are prioritized for diversity coverage and missingness cleanup, not class prediction.",
            ]
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_table_csv}")
    if args.focus == "food_type":
        print(f"Wrote {args.food_queue_csv}")
        print(f"Wrote {args.food_report_md}")


if __name__ == "__main__":
    main()
