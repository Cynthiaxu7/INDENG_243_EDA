from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def plot_model_metric_comparison(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    metrics_long = metrics_df.melt(
        id_vars=["model"],
        value_vars=["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(9.5, 5.2))
    sns.barplot(data=metrics_long, x="metric", y="value", hue="model")
    plt.ylim(0, 1.02)
    plt.title("Model Performance Comparison")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(out_dir / "model_metric_comparison.png")
    plt.close()


def plot_label_distribution(label_df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    ax = sns.barplot(
        data=label_df,
        x="texture_label",
        y="n_samples",
        hue="texture_label",
        palette="Set2",
        legend=False,
    )
    for i, row in label_df.reset_index(drop=True).iterrows():
        ax.text(i, row["n_samples"] + max(1, row["n_samples"] * 0.01), int(row["n_samples"]), ha="center")
    plt.title("Texture Label Distribution")
    plt.xlabel("Texture Label")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(out_dir / "texture_label_distribution.png")
    plt.close()


def plot_video_level_accuracy(pred_df: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    tmp = pred_df.copy()
    tmp["is_correct"] = (tmp["true_label"] == tmp["pred_label"]).astype(int)
    video_acc = (
        tmp.groupby("video_id", as_index=False)["is_correct"]
        .mean()
        .rename(columns={"is_correct": "accuracy"})
        .sort_values("accuracy", ascending=False)
    )
    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=video_acc, x="video_id", y="accuracy", color="#4E79A7")
    plt.ylim(0, 1.02)
    plt.title(f"{model_name} - Accuracy by Video ID")
    plt.xlabel("Video ID")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_accuracy_by_video.png")
    plt.close()


def plot_top_feature_importance(importance_df: pd.DataFrame, out_dir: Path, top_k: int = 15) -> None:
    top = importance_df.sort_values("importance", ascending=False).head(top_k).iloc[::-1]
    plt.figure(figsize=(8.4, 6.0))
    plt.barh(top["feature"], top["importance"], color="#F28E2B")
    plt.title(f"Random Forest Top {top_k} Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_dir / "random_forest_top_features_focus.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create visualization pack for texture model outputs.")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("model_outputs/texture_model"),
        help="Directory containing texture model CSV outputs.",
    )
    args = parser.parse_args()

    model_out = args.model_out
    fig_out = model_out / "figures"
    fig_out.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(model_out / "metrics_summary.csv")
    label_df = pd.read_csv(model_out / "label_distribution.csv")
    rf_importance = pd.read_csv(model_out / "random_forest_feature_importance.csv")
    rf_pred = pd.read_csv(model_out / "random_forest_cv_predictions.csv")
    lr_pred = pd.read_csv(model_out / "logistic_regression_cv_predictions.csv")

    plot_model_metric_comparison(metrics_df, fig_out)
    plot_label_distribution(label_df, fig_out)
    plot_video_level_accuracy(rf_pred, "random_forest", fig_out)
    plot_video_level_accuracy(lr_pred, "logistic_regression", fig_out)
    plot_top_feature_importance(rf_importance, fig_out, top_k=15)

    print(f"Visualization pack created at: {fig_out.resolve()}")


if __name__ == "__main__":
    main()
