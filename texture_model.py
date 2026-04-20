from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def heuristic_texture_label(video_id: str) -> str:
    v = (video_id or "").lower()
    if "steak" in v:
        return "hard"
    if "dessert" in v:
        return "soft"
    if "rameen" in v or "ramen" in v:
        return "medium"
    return "unknown"


def load_label_map(label_map_path: Path) -> Dict[str, str]:
    if label_map_path.exists():
        with label_map_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def aggregate_bite_timeseries(df_ts: pd.DataFrame) -> pd.DataFrame:
    df = df_ts.copy()
    df = df.dropna(subset=["bite_id", "video_id"])
    df["bite_id"] = pd.to_numeric(df["bite_id"], errors="coerce")
    df = df.dropna(subset=["bite_id"])
    df["bite_id"] = df["bite_id"].astype(int)

    numeric_candidates = [
        "mouth_open_px",
        "mouth_width_px",
        "jaw_left_y_px",
        "jaw_right_y_px",
        "mouth_open_smooth",
        "mouth_is_open",
        "phase_open",
        "time_sec",
    ]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    group_cols = ["video_id", "bite_id"]

    aggs = {}
    for c in numeric_cols:
        aggs[c] = ["mean", "std", "min", "max", "median"]

    grouped = df.groupby(group_cols).agg(aggs)
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    grouped = grouped.reset_index()

    if "time_sec" in df.columns:
        dur = (
            df.groupby(group_cols)["time_sec"]
            .agg(["min", "max"])
            .rename(columns={"min": "bite_time_min", "max": "bite_time_max"})
            .reset_index()
        )
        dur["bite_time_span_sec"] = dur["bite_time_max"] - dur["bite_time_min"]
        grouped = grouped.merge(dur[group_cols + ["bite_time_span_sec"]], on=group_cols, how="left")

    return grouped


def build_feature_table(data_dir: Path, label_map: Dict[str, str], drop_unknown: bool) -> pd.DataFrame:
    chewing = pd.read_csv(data_dir / "_ALL_chewing_analysis.csv", low_memory=False)
    timeseries = pd.read_csv(data_dir / "_ALL_mouth_timeseries.csv", low_memory=False)

    chew_cols = [
        "video_id",
        "bite_id",
        "duration_sec",
        "n_chews",
        "chewing_frequency_per_sec",
        "avg_chew_time_sec",
        "mouth_open_frames",
    ]
    chew_cols = [c for c in chew_cols if c in chewing.columns]
    chewing = chewing[chew_cols].copy()

    chewing["bite_id"] = pd.to_numeric(chewing["bite_id"], errors="coerce")
    chewing = chewing.dropna(subset=["bite_id", "video_id"])
    chewing["bite_id"] = chewing["bite_id"].astype(int)

    ts_agg = aggregate_bite_timeseries(timeseries)
    df = chewing.merge(ts_agg, on=["video_id", "bite_id"], how="left")

    df["texture_label"] = df["video_id"].map(label_map)
    missing = df["texture_label"].isna()
    if missing.any():
        df.loc[missing, "texture_label"] = df.loc[missing, "video_id"].map(heuristic_texture_label)

    if drop_unknown:
        df = df[df["texture_label"] != "unknown"].copy()

    return df


def evaluate_with_group_cv(df: pd.DataFrame, out_dir: Path) -> dict:
    y = df["texture_label"].astype(str)
    groups = df["video_id"].astype(str)
    X = df.drop(columns=["texture_label", "video_id"])

    numeric_cols = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        ]
    )

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                Pipeline(
                                    steps=[("imputer", SimpleImputer(strategy="median"))]
                                ),
                                numeric_cols,
                            )
                        ]
                    ),
                ),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)
    if n_splits < 3:
        raise RuntimeError("Need at least 3 unique video_id groups for cross-validation.")

    # Prefer leakage-safe grouped CV. Fallback when some labels are confined to one group.
    labels_per_group = (
        df.groupby("texture_label")["video_id"].nunique().to_dict()
    )
    can_use_group_cv = min(labels_per_group.values()) >= 2
    if can_use_group_cv:
        cv = GroupKFold(n_splits=n_splits)
        cv_groups = groups
        cv_strategy = "GroupKFold(video_id)"
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_groups = None
        cv_strategy = "StratifiedKFold(fallback_no_group)"
        print(
            "WARNING: Using StratifiedKFold fallback because at least one class has <2 unique videos."
        )

    metrics_rows: List[dict] = []
    for model_name, model in models.items():
        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            groups=cv_groups,
            method="predict",
            n_jobs=None,
        )

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "macro_f1": f1_score(y, y_pred, average="macro"),
                "weighted_f1": f1_score(y, y_pred, average="weighted"),
                "n_samples": len(y),
                "n_groups": unique_groups,
                "n_classes": y.nunique(),
                "cv_strategy": cv_strategy,
            }
        )

        labels = sorted(y.unique().tolist())
        cm = confusion_matrix(y, y_pred, labels=labels, normalize="true")
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(out_dir / f"{model_name}_confusion_matrix_normalized.csv")

        plt.figure(figsize=(6.6, 5.4))
        sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.title(f"{model_name} - Normalized Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_confusion_matrix_normalized.png")
        plt.close()

        rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(rep).transpose().to_csv(out_dir / f"{model_name}_classification_report.csv")

        pred_out = pd.DataFrame(
            {
                "video_id": groups.values,
                "true_label": y.values,
                "pred_label": y_pred,
                "model": model_name,
            }
        )
        pred_out.to_csv(out_dir / f"{model_name}_cv_predictions.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("macro_f1", ascending=False)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    # Fit RF on all data for global feature importances
    rf = models["random_forest"]
    rf.fit(X, y)
    rf_clf: RandomForestClassifier = rf.named_steps["clf"]
    importances = pd.DataFrame(
        {"feature": numeric_cols, "importance": rf_clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances.to_csv(out_dir / "random_forest_feature_importance.csv", index=False)

    top = importances.head(20).iloc[::-1]
    plt.figure(figsize=(8.2, 6.8))
    plt.barh(top["feature"], top["importance"], color="#4E79A7")
    plt.title("Random Forest Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_dir / "random_forest_top20_feature_importance.png")
    plt.close()

    return {
        "cv_strategy": cv_strategy,
        "labels_per_unique_video": labels_per_group,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate food-texture models.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing _ALL CSV files.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("texture_label_map.json"),
        help="JSON map of video_id -> texture label.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("model_outputs/texture_model"),
        help="Output directory for metrics and figures.",
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Keep samples with texture label 'unknown' as a class.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(args.label_map)
    df = build_feature_table(
        data_dir=args.data_dir,
        label_map=label_map,
        drop_unknown=not args.keep_unknown,
    )
    if df.empty:
        raise RuntimeError("No samples after label mapping. Update texture_label_map.json.")

    if df["texture_label"].nunique() < 2:
        raise RuntimeError("Need at least 2 texture classes for classification.")

    # Persist assembled modeling table for traceability.
    df.to_csv(out_dir / "modeling_table.csv", index=False)

    summary = (
        df.groupby("texture_label")
        .agg(n_samples=("bite_id", "count"), n_videos=("video_id", "nunique"))
        .reset_index()
    )
    summary.to_csv(out_dir / "label_distribution.csv", index=False)

    cv_info = evaluate_with_group_cv(df=df, out_dir=out_dir)

    run_log = {
        "data_dir": str(args.data_dir),
        "label_map": str(args.label_map),
        "out_dir": str(out_dir),
        "keep_unknown": bool(args.keep_unknown),
        "n_rows": int(len(df)),
        "n_classes": int(df["texture_label"].nunique()),
        "classes": sorted(df["texture_label"].astype(str).unique().tolist()),
        "videos_used": sorted(df["video_id"].astype(str).unique().tolist()),
        "cv_info": cv_info,
    }
    (out_dir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    print("Texture modeling complete.")
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
