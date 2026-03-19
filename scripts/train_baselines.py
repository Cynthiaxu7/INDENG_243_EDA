#!/usr/bin/env python3
"""Train only readiness-approved small-data classification baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

FOOD_ORDER = ["soft", "medium", "hard"]


def _clean_text(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return out


def choose_group_column(df: pd.DataFrame) -> Tuple[str, bool]:
    person = _clean_text(df.get("person_id", pd.Series(dtype=str))).dropna()
    if person.nunique() >= 3:
        return "person_id", False
    return "video_id", True


def load_readiness(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_feature_matrix(df: pd.DataFrame, id_cols: Iterable[str], target: str) -> Tuple[pd.DataFrame, List[str]]:
    drop = set(id_cols) | {target}
    features = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return df[features].copy(), features


def classifier_candidates() -> Dict[str, Pipeline]:
    return {
        "logreg": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "svc_rbf": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, class_weight="balanced")),
            ]
        ),
        "rf": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")),
            ]
        ),
    }


def grouped_splitter(y_enc: np.ndarray, groups: pd.Series, n_splits: int = 5):
    n_groups = groups.nunique()
    n_splits = max(2, min(n_splits, n_groups))
    if len(np.unique(y_enc)) > 1:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return GroupKFold(n_splits=n_splits)


def fit_calibrated(base_model: Pipeline, Xtr: pd.DataFrame, ytr: np.ndarray):
    model = clone(base_model).fit(Xtr, ytr)
    binc = np.bincount(ytr)
    min_class = int(binc[binc > 0].min()) if np.any(binc > 0) else 0
    cal_cv = min(3, min_class)
    if cal_cv >= 2:
        cal = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=cal_cv)
        cal.fit(Xtr, ytr)
        return cal
    return model


def maybe_aggregate_video_level(
    eval_df: pd.DataFrame, classes: np.ndarray, aggregate_to_video: bool
) -> pd.DataFrame:
    if not aggregate_to_video:
        return eval_df
    prob_cols = [f"prob_{c}" for c in classes]
    agg = eval_df.groupby("video_id", as_index=False).agg(
        **{c: (c, "mean") for c in prob_cols},
        y_true=("y_true", "first"),
        n_rows=("y_true", "size"),
    )
    probs = agg[prob_cols].to_numpy()
    pred_idx = probs.argmax(axis=1)
    agg["y_pred"] = [classes[i] for i in pred_idx]
    agg["sample_id"] = agg["video_id"]
    agg["person_id"] = agg.get("person_id", agg["video_id"])
    return agg


def ordinal_probs_logreg(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    y_ord = y_train.map({c: i for i, c in enumerate(FOOD_ORDER)})
    if y_ord.isna().any():
        raise ValueError("food_type contains labels outside [soft, medium, hard]")
    pipes = []
    for thr in [0, 1]:
        y_bin = (y_ord <= thr).astype(int)
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        pipe.fit(X_train, y_bin)
        pipes.append(pipe)
    p_le_soft = pipes[0].predict_proba(X_test)[:, 1]
    p_le_medium = pipes[1].predict_proba(X_test)[:, 1]
    p_soft = np.clip(p_le_soft, 0.0, 1.0)
    p_medium = np.clip(p_le_medium - p_le_soft, 0.0, 1.0)
    p_hard = np.clip(1.0 - p_le_medium, 0.0, 1.0)
    probs = np.vstack([p_soft, p_medium, p_hard]).T
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return probs


def train_classification_task(
    task: str,
    df: pd.DataFrame,
    target_col: str,
    id_cols: List[str],
    models_out: Path,
    aggregate_to_video: bool,
) -> Tuple[List[dict], pd.DataFrame]:
    y_raw = _clean_text(df[target_col])
    d = df.loc[y_raw.notna()].copy()
    if len(d) < 8 or _clean_text(d[target_col]).nunique() < 2:
        return [], pd.DataFrame()

    group_col, provisional = choose_group_column(d)
    d[group_col] = _clean_text(d[group_col]).fillna(_clean_text(d["video_id"])).fillna(_clean_text(d["sample_id"]))
    X, feature_cols = build_feature_matrix(d, id_cols, target_col)
    y = _clean_text(d[target_col])
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    splitter = grouped_splitter(y_enc, d[group_col], n_splits=5)

    results = []
    oof_rows = []
    for model_name, base_model in classifier_candidates().items():
        all_true, all_pred = [], []
        all_proba = []
        total_eval_rows = 0
        total_eval_videos = 0

        for fold, (tr, te) in enumerate(splitter.split(X, y_enc, groups=d[group_col])):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y_enc[tr], y_enc[te]
            fitted = fit_calibrated(base_model, Xtr, ytr)
            pred = fitted.predict(Xte)
            proba = fitted.predict_proba(Xte)

            eval_df = pd.DataFrame(
                {
                    "task": task,
                    "model": model_name,
                    "fold": fold,
                    "sample_id": d.iloc[te]["sample_id"].astype(str).values,
                    "video_id": d.iloc[te]["video_id"].astype(str).values,
                    "person_id": _clean_text(d.iloc[te]["person_id"]).fillna(d.iloc[te]["video_id"]).values,
                    "y_true": le.inverse_transform(yte),
                    "y_pred": le.inverse_transform(pred),
                }
            )
            for cidx, cname in enumerate(le.classes_):
                eval_df[f"prob_{cname}"] = proba[:, cidx]
            eval_df = maybe_aggregate_video_level(eval_df, le.classes_, aggregate_to_video)
            total_eval_rows += len(te)
            total_eval_videos += eval_df["video_id"].nunique()

            all_true.extend(eval_df["y_true"].tolist())
            all_pred.extend(eval_df["y_pred"].tolist())
            all_proba.append(eval_df[[f"prob_{c}" for c in le.classes_]].to_numpy())
            oof_rows.extend(eval_df.to_dict(orient="records"))

        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        row = {
            "task": task,
            "model": model_name,
            "metric_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "metric_weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "n_eval_samples": int(len(y_true)),
            "n_input_rows": int(total_eval_rows),
            "n_eval_videos": int(total_eval_videos),
            "group_col": group_col,
            "provisional_grouping": bool(provisional),
            "evaluation_level": "video" if aggregate_to_video else "sample",
        }
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=le.classes_, zero_division=0
        )
        for i, cname in enumerate(le.classes_):
            row[f"{cname}_precision"] = float(prec[i])
            row[f"{cname}_recall"] = float(rec[i])
            row[f"{cname}_f1"] = float(f1[i])
            row[f"{cname}_support"] = int(sup[i])
        try:
            proba_all = np.vstack(all_proba)
            if proba_all.shape[1] == 2:
                row["metric_roc_auc"] = float(roc_auc_score((y_true == le.classes_[1]).astype(int), proba_all[:, 1]))
            elif proba_all.shape[1] > 2:
                y_true_idx = np.array([np.where(le.classes_ == v)[0][0] for v in y_true])
                row["metric_roc_auc"] = float(
                    roc_auc_score(y_true_idx, proba_all, multi_class="ovr", average="macro")
                )
        except Exception:
            row["metric_roc_auc"] = np.nan
        row["classification_report_json"] = json.dumps(
            classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        )
        results.append(row)

        final = fit_calibrated(base_model, X, y_enc)
        joblib.dump({"model": final, "label_encoder": le, "feature_cols": feature_cols}, models_out / f"{task}__{model_name}.joblib")

    if task == "food_type":
        y_true_all, y_pred_all = [], []
        for fold, (tr, te) in enumerate(splitter.split(X, y_enc, groups=d[group_col])):
            probs = ordinal_probs_logreg(X.iloc[tr], y.iloc[tr], X.iloc[te])
            pred_idx = probs.argmax(axis=1)
            eval_df = pd.DataFrame(
                {
                    "task": task,
                    "model": "ordinal_logreg",
                    "fold": fold,
                    "sample_id": d.iloc[te]["sample_id"].astype(str).values,
                    "video_id": d.iloc[te]["video_id"].astype(str).values,
                    "person_id": _clean_text(d.iloc[te]["person_id"]).fillna(d.iloc[te]["video_id"]).values,
                    "y_true": y.iloc[te].values,
                    "y_pred": [FOOD_ORDER[i] for i in pred_idx],
                    "prob_soft": probs[:, 0],
                    "prob_medium": probs[:, 1],
                    "prob_hard": probs[:, 2],
                }
            )
            eval_df = maybe_aggregate_video_level(eval_df, np.array(FOOD_ORDER), aggregate_to_video)
            y_true_all.extend(eval_df["y_true"].tolist())
            y_pred_all.extend(eval_df["y_pred"].tolist())
            oof_rows.extend(eval_df.to_dict(orient="records"))
        results.append(
            {
                "task": task,
                "model": "ordinal_logreg",
                "metric_macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro")),
                "metric_weighted_f1": float(f1_score(y_true_all, y_pred_all, average="weighted")),
                "group_col": group_col,
                "provisional_grouping": bool(provisional),
                "evaluation_level": "video" if aggregate_to_video else "sample",
            }
        )
    return results, pd.DataFrame(oof_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train readiness-gated grouped-CV classification baselines.")
    parser.add_argument("--features-csv", type=Path, default=Path("artifacts/features/video_level_features.csv"))
    parser.add_argument("--window-features-csv", type=Path, default=Path("artifacts/features/window_level_features.csv"))
    parser.add_argument("--feature-level", choices=["video", "window"], default="video")
    parser.add_argument("--labels-csv", type=Path, default=Path("metadata/labels_template.csv"))
    parser.add_argument("--readiness-json", type=Path, default=Path("reports/task_readiness_report.json"))
    parser.add_argument("--results-csv", type=Path, default=Path("artifacts/results/cv_metrics.csv"))
    parser.add_argument("--predictions-csv", type=Path, default=Path("artifacts/results/oof_predictions.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--allow-secondary-tasks", action="store_true")
    args = parser.parse_args()

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    feats_path = args.features_csv if args.feature_level == "video" else args.window_features_csv
    if not feats_path.exists():
        raise SystemExit(f"Missing features CSV: {feats_path}")
    if not args.labels_csv.exists():
        raise SystemExit(f"Missing labels CSV: {args.labels_csv}")

    feat = pd.read_csv(feats_path)
    labels = pd.read_csv(args.labels_csv)
    readiness = load_readiness(args.readiness_json).get("tasks", {})

    feat["sample_id"] = feat["sample_id"].astype(str)
    labels["sample_id"] = labels["sample_id"].astype(str)

    if "chew_side_label" not in labels.columns and "chewing_side_dominance" in labels.columns:
        labels["chew_side_label"] = labels["chewing_side_dominance"]

    data = feat.merge(labels, on="sample_id", how="left", suffixes=("", "_label"))
    for col in ["sample_id", "video_id", "person_id"]:
        if col not in data.columns:
            data[col] = data.get(f"{col}_label", "")
        data[col] = data[col].astype(str)

    tasks = [("food_type", True), ("chew_side_label", False), ("rhythm_class", False), ("pause_style", False)]
    id_cols = ["sample_id", "video_id", "person_id", "window_id", "data_path"]

    all_results, all_preds, trained = [], [], []
    for task, primary in tasks:
        if task not in data.columns:
            continue
        trainable = readiness.get(task, {}).get("trainable_now", False)
        if not trainable:
            continue
        if not primary and not args.allow_secondary_tasks and task != "chew_side_label":
            continue
        res, preds = train_classification_task(
            task=task,
            df=data,
            target_col=task,
            id_cols=id_cols,
            models_out=args.models_dir,
            aggregate_to_video=(args.feature_level == "window"),
        )
        if res:
            all_results.extend(res)
            trained.append(task)
        if not preds.empty:
            all_preds.append(preds)

    if not all_results:
        raise SystemExit("No trainable tasks passed readiness checks. Fill labels and rerun readiness.")

    pd.DataFrame(all_results).to_csv(args.results_csv, index=False)
    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    pred_df.to_csv(args.predictions_csv, index=False)
    print(f"Trained tasks: {trained}")
    print(f"Wrote CV metrics: {args.results_csv}")
    print(f"Wrote OOF predictions: {args.predictions_csv}")


if __name__ == "__main__":
    main()
