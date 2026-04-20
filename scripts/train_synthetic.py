#!/usr/bin/env python3
"""Train window-level food-type classifiers on synthetic low/med/high noise datasets.

Models:
  logreg / rf / xgb   - classical + GBM tabular
  mlp / ftt           - PyTorch tabular NN (MLP and FT-Transformer via rtdl)
  mamba               - hand-written bidirectional Mamba-1 + Mamba-3 upgrades
                        (complex-valued A + trapezoidal discretization)
                        trained on per-video window sequences
  mamba_ssm_v1        - Tri Dao's fused-kernel Mamba-1 reference baseline
                        (real-diagonal A + Euler) — same bidirectional wrapper

CV: StratifiedGroupKFold(5) on video_id (fallback to GroupKFold).
Metrics per (noise_level, model): per-window macro-F1 and per-chunk
majority-vote macro-F1 computed from OOF predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
# xgboost must be imported BEFORE torch on macOS to avoid libomp conflict SIGSEGV.
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from rtdl_revisiting_models import FTTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.mamba_hybrid import WindowSequenceClassifier  # noqa: E402
from models.mamba_ssm_baseline import MambaSSMv1Classifier  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "artifacts" / "features"
RESULTS_DIR = REPO_ROOT / "artifacts" / "results"

NOISE_LEVELS: List[str] = ["low", "med", "high"]
CLASS_ORDER: List[str] = ["soft", "medium", "hard"]
GROUP_COL = "video_id"
TARGET_COL = "food_type_synthetic"
WINDOW_ID_COL = "window_id"
CHUNK_COL = "chunk_id"
N_SPLITS = 5
RANDOM_STATE = 42
EXPECTED_N_ROWS = 8077
EXPECTED_N_VIDEOS = 20

ID_COLS: List[str] = [
    "sample_id", "video_id", "person_id", "window_id", "data_path",
    "chunk_id", "food_type_synthetic", "window_type", "bite_id",
]
FORBIDDEN_FEATURES = {
    "chunk_id", "food_type_synthetic", "sample_id", "video_id", "person_id", "window_id",
}

DEVICE = "cpu"

# NN tabular hyperparams
NN_EPOCHS = 60
NN_BATCH_SIZE = 256
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4

# Mamba hyperparams — defaults sized for a GPU. CPU runs should pass
# `--mamba-d-model 32 --mamba-n-blocks 1 --mamba-d-state 8 --mamba-epochs 25`
# via the CLI to keep per-epoch time tractable (< 15 s/epoch).
MAMBA_EPOCHS = 80
MAMBA_BATCH_SIZE = 4
MAMBA_LR = 5e-4
MAMBA_WEIGHT_DECAY = 1e-3
MAMBA_D_MODEL = 64
MAMBA_N_BLOCKS = 2
MAMBA_D_STATE = 16
MAMBA_LOG_EVERY = 5


# ---------------------------------------------------------------------------
# Data loading & feature selection
# ---------------------------------------------------------------------------

def load_synthetic(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing synthetic features CSV: {path}")
    df = pd.read_csv(path)
    if len(df) != EXPECTED_N_ROWS:
        raise ValueError(f"{path.name}: expected {EXPECTED_N_ROWS} rows, got {len(df)}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"{path.name}: missing target column {TARGET_COL}")
    bad = set(df[TARGET_COL].dropna().unique()) - set(CLASS_ORDER)
    if bad:
        raise ValueError(f"{path.name}: unexpected labels in {TARGET_COL}: {sorted(bad)}")
    if df[GROUP_COL].nunique() != EXPECTED_N_VIDEOS:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_N_VIDEOS} unique {GROUP_COL} values, got {df[GROUP_COL].nunique()}"
        )
    return df


def select_features(df: pd.DataFrame) -> List[str]:
    drop = set(ID_COLS)
    feats = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    leaked = [c for c in feats if c in FORBIDDEN_FEATURES]
    if leaked:
        raise ValueError(f"Feature list contains forbidden id/target columns: {leaked}")
    if not feats:
        raise ValueError("No numeric feature columns selected after filtering")
    return feats


def impute_features_inplace(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """Fill NaN in feature columns with column median; return the fill values."""
    fills: Dict[str, float] = {}
    for col in feature_cols:
        if df[col].isna().any():
            med = float(df[col].median())
            df[col] = df[col].fillna(med)
            fills[col] = med
    return fills


def make_splitter(y_enc: np.ndarray, groups: pd.Series) -> Tuple[object, str]:
    try:
        sgk = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(sgk.split(np.zeros(len(y_enc)), y_enc, groups=groups))
        if len(splits) != N_SPLITS:
            raise ValueError("StratifiedGroupKFold produced unexpected number of splits")
        return sgk, "StratifiedGroupKFold"
    except Exception as exc:
        print(f"[warn] StratifiedGroupKFold failed ({exc!r}); falling back to GroupKFold")
        return GroupKFold(n_splits=N_SPLITS), "GroupKFold"


# ---------------------------------------------------------------------------
# PyTorch tabular wrapper (sklearn-compatible)
# ---------------------------------------------------------------------------

class TorchTabularClassifier:
    """Thin sklearn-style wrapper around a PyTorch tabular classifier."""

    def __init__(
        self,
        net_factory: Callable[[int, int], nn.Module],
        forward_style: str = "single",
        epochs: int | None = None,
        batch_size: int = NN_BATCH_SIZE,
        lr: float = NN_LR,
        weight_decay: float = NN_WEIGHT_DECAY,
        device: str | None = None,
        seed: int = RANDOM_STATE,
    ):
        # Resolve at instantiation time so CLI overrides of module-level
        # globals are picked up.
        self.net_factory = net_factory
        self.forward_style = forward_style
        self.epochs = epochs if epochs is not None else NN_EPOCHS
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device if device is not None else DEVICE
        self.seed = seed
        self.net: nn.Module | None = None
        self.classes_: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _fwd(self, X: torch.Tensor) -> torch.Tensor:
        if self.forward_style == "ftt":
            return self.net(X, None)
        return self.net(X)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y)
        classes = np.array(sorted(np.unique(y_arr).tolist()))
        self.classes_ = classes

        mean = X_arr.mean(axis=0)
        std = X_arr.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        self._mean = mean.astype(np.float32)
        self._std = std
        Xn = (X_arr - self._mean) / self._std

        cw = compute_class_weight("balanced", classes=classes, y=y_arr)
        cw_t = torch.tensor(cw, dtype=torch.float32, device=self.device)

        d_in = Xn.shape[1]
        n_classes = len(classes)
        self.net = self.net_factory(d_in, n_classes).to(self.device)

        cls_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([cls_to_idx[v] for v in y_arr], dtype=np.int64)

        opt = AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.epochs)

        X_t = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_idx, dtype=torch.long, device=self.device)
        n = X_t.shape[0]

        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                Xb = X_t[idx]
                yb = y_t[idx]
                logits = self._fwd(Xb)
                loss = F.cross_entropy(logits, yb, weight=cw_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        Xn = (X_arr - self._mean) / self._std
        X_t = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        self.net.eval()
        with torch.no_grad():
            logits = self._fwd(X_t)
            proba = F.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def predict(self, X) -> np.ndarray:
        idx = self.predict_proba(X).argmax(axis=1)
        return self.classes_[idx]


def mlp_factory(d_in: int, n_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, n_classes),
    )


def ftt_factory(d_in: int, n_classes: int) -> nn.Module:
    return FTTransformer(
        n_cont_features=d_in, cat_cardinalities=[],
        n_blocks=2, d_block=64, attention_n_heads=4,
        attention_dropout=0.1, ffn_d_hidden_multiplier=1.33, ffn_dropout=0.1,
        residual_dropout=0.0, d_out=n_classes, _is_default=True,
    )


def build_tabular_models() -> Dict[str, object]:
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=len(CLASS_ORDER),
            eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE,
            tree_method="hist",
        ),
        "mlp": TorchTabularClassifier(mlp_factory, forward_style="single"),
        "ftt": TorchTabularClassifier(ftt_factory, forward_style="ftt", epochs=max(1, NN_EPOCHS // 2)),
    }


def fresh_tabular(name: str) -> object:
    return build_tabular_models()[name]


def global_importance(name: str, estimator: object, feature_cols: List[str]) -> np.ndarray | None:
    """Returns importance vector or None when model has no natural importance."""
    if name == "logreg":
        clf: LogisticRegression = estimator.named_steps["clf"]  # type: ignore[assignment]
        return np.abs(np.asarray(clf.coef_)).mean(axis=0)
    if name in ("rf", "xgb"):
        return np.asarray(getattr(estimator, "feature_importances_"))
    return None  # MLP / FTT / Mamba importances computed via permutation elsewhere if needed


# ---------------------------------------------------------------------------
# Mamba CV path
# ---------------------------------------------------------------------------

def _video_sequences(df: pd.DataFrame, idx: np.ndarray, feature_cols: List[str],
                     mean: np.ndarray, std: np.ndarray
                     ) -> Tuple[List[np.ndarray], List[np.ndarray], pd.DataFrame]:
    sub = df.iloc[idx].sort_values([GROUP_COL, WINDOW_ID_COL])
    X_seqs, y_seqs, metas = [], [], []
    for vid, g in sub.groupby(GROUP_COL, sort=True):
        X = (g[feature_cols].to_numpy(dtype=np.float32) - mean) / std
        y = g[TARGET_COL].to_numpy()
        X_seqs.append(X); y_seqs.append(y)
        metas.append(g[["sample_id", "video_id", "window_id", "chunk_id"]].copy())
    meta_df = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    return X_seqs, y_seqs, meta_df


def run_mamba_fold(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold: int,
    model_cls: Callable[..., nn.Module] = WindowSequenceClassifier,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    torch.manual_seed(RANDOM_STATE + fold)

    train_rows = df.iloc[train_idx]
    mean = train_rows[feature_cols].mean().to_numpy(dtype=np.float32)
    std = train_rows[feature_cols].std().to_numpy(dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    X_tr, y_tr, _ = _video_sequences(df, train_idx, feature_cols, mean, std)
    X_va, y_va, meta_va = _video_sequences(df, val_idx, feature_cols, mean, std)

    cls_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_tr_idx = [np.array([cls_to_idx[v] for v in y], dtype=np.int64) for y in y_tr]

    # class weights from train pool
    all_train_labels = np.concatenate(y_tr)
    cw = compute_class_weight("balanced", classes=np.array(CLASS_ORDER), y=all_train_labels)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

    clf = model_cls(
        d_features=len(feature_cols),
        d_model=MAMBA_D_MODEL,
        n_blocks=MAMBA_N_BLOCKS,
        d_state=MAMBA_D_STATE,
        n_classes=len(CLASS_ORDER),
    ).to(DEVICE)
    opt = AdamW(clf.parameters(), lr=MAMBA_LR, weight_decay=MAMBA_WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=MAMBA_EPOCHS)

    train_X_t = [torch.tensor(x, dtype=torch.float32, device=DEVICE) for x in X_tr]
    train_y_t = [torch.tensor(y, dtype=torch.long, device=DEVICE) for y in y_tr_idx]
    n_train = len(train_X_t)

    for _ in range(MAMBA_EPOCHS):
        clf.train()
        perm = torch.randperm(n_train).tolist()
        for i in range(0, n_train, MAMBA_BATCH_SIZE):
            batch = perm[i:i + MAMBA_BATCH_SIZE]
            lens = [train_X_t[j].shape[0] for j in batch]
            Tmax = max(lens)
            D = train_X_t[batch[0]].shape[1]
            Xb = torch.zeros(len(batch), Tmax, D, device=DEVICE)
            yb = torch.full((len(batch), Tmax), -100, dtype=torch.long, device=DEVICE)
            for k, j in enumerate(batch):
                Xb[k, :lens[k]] = train_X_t[j]
                yb[k, :lens[k]] = train_y_t[j]
            logits = clf(Xb)
            loss = F.cross_entropy(
                logits.reshape(-1, len(CLASS_ORDER)),
                yb.reshape(-1),
                weight=cw_t,
                ignore_index=-100,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    clf.eval()
    y_true_all: List[str] = []
    y_pred_all: List[str] = []
    prob_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for X, y in zip(X_va, y_va):
            Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = clf(Xt)
            proba = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_idx = proba.argmax(axis=1)
            y_pred_all.extend(CLASS_ORDER[i] for i in pred_idx)
            y_true_all.extend(y.tolist())
            prob_chunks.append(proba)
    prob_arr = np.concatenate(prob_chunks, axis=0) if prob_chunks else np.zeros((0, len(CLASS_ORDER)))
    return np.array(y_true_all), np.array(y_pred_all), prob_arr, meta_va


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    row: Dict[str, float] = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0)
        ),
    }
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    for i, cname in enumerate(CLASS_ORDER):
        row[f"{cname}_precision"] = float(prec[i])
        row[f"{cname}_recall"] = float(rec[i])
        row[f"{cname}_f1"] = float(f1[i])
        row[f"{cname}_support"] = int(sup[i])
    return row


def chunk_level_metrics(oof_df: pd.DataFrame) -> Dict[str, float]:
    """Majority-vote aggregation of OOF per-window predictions to chunk level."""
    if oof_df.empty:
        return {"chunk_macro_f1": float("nan"), "chunk_weighted_f1": float("nan"), "chunk_acc": float("nan"), "n_chunks": 0}
    agg = oof_df.groupby(CHUNK_COL).agg(
        y_pred=("y_pred", lambda s: s.mode().iloc[0]),
        y_true=("y_true", "first"),
    )
    return {
        "chunk_macro_f1": float(
            f1_score(agg["y_true"], agg["y_pred"], labels=CLASS_ORDER, average="macro", zero_division=0)
        ),
        "chunk_weighted_f1": float(
            f1_score(agg["y_true"], agg["y_pred"], labels=CLASS_ORDER, average="weighted", zero_division=0)
        ),
        "chunk_acc": float((agg["y_true"] == agg["y_pred"]).mean()),
        "n_chunks": int(len(agg)),
    }


def aggregate_numeric_mean(rows: List[dict], numeric_cols: List[str]) -> dict:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return {col: float(df[col].mean()) for col in numeric_cols if col in df.columns}


# ---------------------------------------------------------------------------
# Main per-noise-level routine
# ---------------------------------------------------------------------------

def run_noise_level(
    noise: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    active_models: List[str],
) -> Tuple[List[dict], List[dict], List[dict], Dict[str, dict], str, float]:
    t_start = time.time()
    y_str = df[TARGET_COL].astype(str).to_numpy()
    le = LabelEncoder().fit(CLASS_ORDER)
    y_enc = le.transform(y_str)
    groups = df[GROUP_COL].astype(str)
    X = df[feature_cols].copy()

    splitter, cv_name = make_splitter(y_enc, groups)

    metric_rows: List[dict] = []
    oof_rows: List[dict] = []
    importance_rows: List[dict] = []
    confusion_per_model: Dict[str, dict] = {}

    # -- Tabular models --
    tabular_models = {k: v for k, v in build_tabular_models().items() if k in active_models}
    for model_name in tabular_models:
        t_m = time.time()
        fold_rows: List[dict] = []
        y_true_full: List[str] = []
        y_pred_full: List[str] = []
        model_oof: List[dict] = []

        for fold, (tr, te) in enumerate(splitter.split(X, y_enc, groups=groups)):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y_enc[tr], y_enc[te]

            est = fresh_tabular(model_name)
            # sklearn and xgb accept encoded ints; our NN wrappers want class strings.
            if isinstance(est, TorchTabularClassifier):
                y_tr_fit = le.inverse_transform(ytr)
            else:
                y_tr_fit = ytr
            est.fit(Xtr, y_tr_fit)

            if isinstance(est, TorchTabularClassifier):
                proba = est.predict_proba(Xte)
                pred = est.predict(Xte)
                y_pred_str = np.asarray(pred)
                classes_ = list(est.classes_)
            else:
                proba = est.predict_proba(Xte)
                pred_enc = est.predict(Xte)
                y_pred_str = le.inverse_transform(np.asarray(pred_enc).astype(int))
                classes_ = list(le.inverse_transform(np.asarray(est.classes_, dtype=int)))

            class_to_col = {c: i for i, c in enumerate(classes_)}
            proba_ordered = np.zeros((proba.shape[0], len(CLASS_ORDER)), dtype=float)
            for i, cname in enumerate(CLASS_ORDER):
                if cname in class_to_col:
                    proba_ordered[:, i] = proba[:, class_to_col[cname]]

            y_true_str = le.inverse_transform(yte)
            row = {
                "noise_level": noise,
                "model": model_name,
                "fold": fold,
                **per_fold_metrics(y_true_str, y_pred_str),
                "n_train": int(len(tr)),
                "n_val": int(len(te)),
                "train_videos": int(groups.iloc[tr].nunique()),
                "val_videos": int(groups.iloc[te].nunique()),
            }
            fold_rows.append(row)
            metric_rows.append(row)
            y_true_full.extend(y_true_str.tolist())
            y_pred_full.extend(y_pred_str.tolist())

            val_df = df.iloc[te].reset_index(drop=True)
            for j in range(len(val_df)):
                model_oof.append(
                    {
                        "noise_level": noise, "model": model_name, "fold": fold,
                        "sample_id": val_df.loc[j, "sample_id"],
                        "video_id": val_df.loc[j, "video_id"],
                        "window_id": val_df.loc[j, "window_id"],
                        "chunk_id": val_df.loc[j, "chunk_id"],
                        "y_true": y_true_str[j],
                        "y_pred": y_pred_str[j],
                        "prob_soft": float(proba_ordered[j, 0]),
                        "prob_medium": float(proba_ordered[j, 1]),
                        "prob_hard": float(proba_ordered[j, 2]),
                    }
                )

        oof_rows.extend(model_oof)

        # fold-mean aggregate + chunk-level metrics
        numeric_cols = [
            "macro_f1", "weighted_f1",
            "soft_precision", "soft_recall", "soft_f1", "soft_support",
            "medium_precision", "medium_recall", "medium_f1", "medium_support",
            "hard_precision", "hard_recall", "hard_f1", "hard_support",
            "n_train", "n_val", "train_videos", "val_videos",
        ]
        chunk_m = chunk_level_metrics(pd.DataFrame(model_oof))
        mean_row = {
            "noise_level": noise, "model": model_name, "fold": "mean",
            **aggregate_numeric_mean(fold_rows, numeric_cols),
            **chunk_m,
        }
        metric_rows.append(mean_row)

        cm = confusion_matrix(y_true_full, y_pred_full, labels=CLASS_ORDER)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = np.where(row_sums > 0, cm / np.maximum(row_sums, 1), 0.0)
        confusion_per_model[model_name] = {
            "labels": list(CLASS_ORDER),
            "matrix": cm.astype(int).tolist(),
            "normalized": cm_norm.astype(float).tolist(),
        }

        # global feature importance — refit once on all data
        imp = None
        if model_name in ("logreg", "rf", "xgb"):
            final_est = fresh_tabular(model_name)
            if isinstance(final_est, TorchTabularClassifier):
                final_est.fit(X, y_str)
            else:
                final_est.fit(X, y_enc)
            imp = global_importance(model_name, final_est, feature_cols)
        if imp is not None:
            for fname, val in zip(feature_cols, imp):
                importance_rows.append(
                    {"noise_level": noise, "model": model_name, "feature": fname, "importance": float(val)}
                )

        print(f"  [{noise}] {model_name} done in {time.time()-t_m:.1f}s  window-F1={mean_row['macro_f1']:.3f}  chunk-F1={mean_row.get('chunk_macro_f1', float('nan')):.3f}")

    mamba_variants: List[Tuple[str, Callable[..., nn.Module]]] = [
        ("mamba", WindowSequenceClassifier),
        ("mamba_ssm_v1", MambaSSMv1Classifier),
    ]
    active_mamba = [(name, cls) for name, cls in mamba_variants if name in active_models]
    if not active_mamba:
        elapsed = time.time() - t_start
        return metric_rows, oof_rows, importance_rows, confusion_per_model, cv_name, elapsed

    # -- Mamba sequence model(s) --
    for variant_name, variant_cls in active_mamba:
        t_m = time.time()
        fold_rows = []
        y_true_full = []
        y_pred_full = []
        mamba_oof: List[dict] = []
        for fold, (tr, te) in enumerate(splitter.split(X, y_enc, groups=groups)):
            y_true_f, y_pred_f, proba_f, meta_f = run_mamba_fold(
                df, feature_cols, tr, te, fold, model_cls=variant_cls,
            )
            row = {
                "noise_level": noise,
                "model": variant_name,
                "fold": fold,
                **per_fold_metrics(y_true_f, y_pred_f),
                "n_train": int(len(tr)),
                "n_val": int(len(te)),
                "train_videos": int(groups.iloc[tr].nunique()),
                "val_videos": int(groups.iloc[te].nunique()),
            }
            fold_rows.append(row)
            metric_rows.append(row)
            y_true_full.extend(y_true_f.tolist())
            y_pred_full.extend(y_pred_f.tolist())
            for j in range(len(meta_f)):
                mamba_oof.append(
                    {
                        "noise_level": noise, "model": variant_name, "fold": fold,
                        "sample_id": meta_f.loc[j, "sample_id"],
                        "video_id": meta_f.loc[j, "video_id"],
                        "window_id": meta_f.loc[j, "window_id"],
                        "chunk_id": meta_f.loc[j, "chunk_id"],
                        "y_true": y_true_f[j],
                        "y_pred": y_pred_f[j],
                        "prob_soft": float(proba_f[j, 0]),
                        "prob_medium": float(proba_f[j, 1]),
                        "prob_hard": float(proba_f[j, 2]),
                    }
                )
        oof_rows.extend(mamba_oof)

        chunk_m = chunk_level_metrics(pd.DataFrame(mamba_oof))
        numeric_cols = [
            "macro_f1", "weighted_f1",
            "soft_precision", "soft_recall", "soft_f1", "soft_support",
            "medium_precision", "medium_recall", "medium_f1", "medium_support",
            "hard_precision", "hard_recall", "hard_f1", "hard_support",
            "n_train", "n_val", "train_videos", "val_videos",
        ]
        mean_row = {
            "noise_level": noise, "model": variant_name, "fold": "mean",
            **aggregate_numeric_mean(fold_rows, numeric_cols),
            **chunk_m,
        }
        metric_rows.append(mean_row)

        cm = confusion_matrix(y_true_full, y_pred_full, labels=CLASS_ORDER)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = np.where(row_sums > 0, cm / np.maximum(row_sums, 1), 0.0)
        confusion_per_model[variant_name] = {
            "labels": list(CLASS_ORDER),
            "matrix": cm.astype(int).tolist(),
            "normalized": cm_norm.astype(float).tolist(),
        }

        print(f"  [{noise}] {variant_name} done in {time.time()-t_m:.1f}s  window-F1={mean_row['macro_f1']:.3f}  chunk-F1={mean_row.get('chunk_macro_f1', float('nan')):.3f}")

    elapsed = time.time() - t_start
    return metric_rows, oof_rows, importance_rows, confusion_per_model, cv_name, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train synthetic-texture classifiers.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="Device for torch models (tabular NNs + Mamba). sklearn/xgb always on CPU.")
    p.add_argument("--models", type=str, default="logreg,rf,xgb,mlp,ftt,mamba,mamba_ssm_v1",
                   help="Comma-separated subset of {logreg,rf,xgb,mlp,ftt,mamba,mamba_ssm_v1}.")
    p.add_argument("--noise-levels", type=str, default="low,med,high",
                   help="Comma-separated subset of {low,med,high}.")
    p.add_argument("--mamba-epochs", type=int, default=MAMBA_EPOCHS)
    p.add_argument("--mamba-d-model", type=int, default=MAMBA_D_MODEL)
    p.add_argument("--mamba-n-blocks", type=int, default=MAMBA_N_BLOCKS)
    p.add_argument("--mamba-d-state", type=int, default=MAMBA_D_STATE)
    p.add_argument("--mamba-lr", type=float, default=MAMBA_LR)
    p.add_argument("--nn-epochs", type=int, default=NN_EPOCHS)
    p.add_argument("--results-suffix", type=str, default="",
                   help="Optional suffix appended to output filenames (e.g. '_gpu' for partial runs).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # override module-level config from CLI
    global DEVICE, MAMBA_EPOCHS, MAMBA_D_MODEL, MAMBA_N_BLOCKS, MAMBA_D_STATE, MAMBA_LR, NN_EPOCHS
    DEVICE = args.device
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but torch.cuda.is_available()=False; falling back to cpu")
        DEVICE = "cpu"
    MAMBA_EPOCHS = args.mamba_epochs
    MAMBA_D_MODEL = args.mamba_d_model
    MAMBA_N_BLOCKS = args.mamba_n_blocks
    MAMBA_D_STATE = args.mamba_d_state
    MAMBA_LR = args.mamba_lr
    NN_EPOCHS = args.nn_epochs

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    valid_models = {"logreg", "rf", "xgb", "mlp", "ftt", "mamba", "mamba_ssm_v1"}
    bad = [m for m in requested_models if m not in valid_models]
    if bad:
        raise ValueError(f"Unknown models: {bad}. Valid: {sorted(valid_models)}")

    requested_noise = [n.strip() for n in args.noise_levels.split(",") if n.strip()]
    valid_noise = {"low", "med", "high"}
    bad_n = [n for n in requested_noise if n not in valid_noise]
    if bad_n:
        raise ValueError(f"Unknown noise levels: {bad_n}. Valid: {sorted(valid_noise)}")

    print(f"[cfg] device={DEVICE}  models={requested_models}  noise_levels={requested_noise}")
    print(f"[cfg] mamba d_model={MAMBA_D_MODEL} n_blocks={MAMBA_N_BLOCKS} d_state={MAMBA_D_STATE} epochs={MAMBA_EPOCHS}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    all_metrics: List[dict] = []
    all_oof: List[dict] = []
    all_importance: List[dict] = []
    cm_all: Dict[str, dict] = {}
    per_noise_seconds: Dict[str, float] = {}
    feature_cols_used: List[str] | None = None
    cv_strategy_seen: str | None = None

    imputation_fills_first: Dict[str, float] | None = None
    for noise in requested_noise:
        print(f"\n=== Noise level: {noise} ===")
        path = FEATURES_DIR / f"window_level_features_synthetic_{noise}.csv"
        df = load_synthetic(path)
        feats = select_features(df)
        fills = impute_features_inplace(df, feats)
        if imputation_fills_first is None:
            imputation_fills_first = fills
        if fills:
            print(f"  imputed NaN in {len(fills)} cols (inherited from raw extraction)")
        if feature_cols_used is None:
            feature_cols_used = feats
        elif feature_cols_used != feats:
            raise ValueError(f"Feature columns differ across noise levels: {noise} vs first observed")

        metrics, oof, importance, cm, cv_name, elapsed = run_noise_level(noise, df, feats, requested_models)
        all_metrics.extend(metrics)
        all_oof.extend(oof)
        all_importance.extend(importance)
        cm_all[noise] = cm
        per_noise_seconds[noise] = elapsed
        if cv_strategy_seen is None:
            cv_strategy_seen = cv_name
        elif cv_strategy_seen != cv_name:
            cv_strategy_seen = "GroupKFold"

    # -- Write artifacts --
    metrics_cols = [
        "noise_level", "model", "fold",
        "macro_f1", "weighted_f1",
        "soft_precision", "soft_recall", "soft_f1", "soft_support",
        "medium_precision", "medium_recall", "medium_f1", "medium_support",
        "hard_precision", "hard_recall", "hard_f1", "hard_support",
        "n_train", "n_val", "train_videos", "val_videos",
        "chunk_macro_f1", "chunk_weighted_f1", "chunk_acc", "n_chunks",
    ]
    metrics_df = pd.DataFrame(all_metrics)
    for col in metrics_cols:
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan
    metrics_df = metrics_df[metrics_cols]
    sfx = args.results_suffix
    metrics_df.to_csv(RESULTS_DIR / f"synthetic_cv_metrics{sfx}.csv", index=False)

    oof_cols = [
        "noise_level", "model", "fold", "sample_id", "video_id", "window_id", "chunk_id",
        "y_true", "y_pred", "prob_soft", "prob_medium", "prob_hard",
    ]
    oof_df = pd.DataFrame(all_oof)
    for col in oof_cols:
        if col not in oof_df.columns:
            oof_df[col] = np.nan
    oof_df = oof_df[oof_cols]
    oof_df.to_csv(RESULTS_DIR / f"synthetic_oof_predictions{sfx}.csv", index=False)

    imp_df = pd.DataFrame(all_importance, columns=["noise_level", "model", "feature", "importance"])
    imp_df.to_csv(RESULTS_DIR / f"synthetic_feature_importance{sfx}.csv", index=False)

    (RESULTS_DIR / f"synthetic_confusion_matrices{sfx}.json").write_text(
        json.dumps(cm_all, indent=2), encoding="utf-8"
    )

    run_log = {
        "cv_strategy": cv_strategy_seen or "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "group_column": GROUP_COL,
        "random_state": RANDOM_STATE,
        "n_videos": EXPECTED_N_VIDEOS,
        "n_windows": EXPECTED_N_ROWS,
        "noise_levels": list(requested_noise),
        "models": list(requested_models),
        "device": DEVICE,
        "tabular_nn_config": {
            "epochs": NN_EPOCHS, "batch_size": NN_BATCH_SIZE,
            "lr": NN_LR, "weight_decay": NN_WEIGHT_DECAY,
        },
        "mamba_config": {
            "d_model": MAMBA_D_MODEL, "n_blocks": MAMBA_N_BLOCKS, "d_state": MAMBA_D_STATE,
            "epochs": MAMBA_EPOCHS, "batch_size": MAMBA_BATCH_SIZE,
            "lr": MAMBA_LR, "weight_decay": MAMBA_WEIGHT_DECAY,
            "upgrades": ["complex_valued_A (Mamba-3)", "trapezoidal_discretization (Mamba-3)", "bidirectional"],
            "skipped": ["MIMO_SSM (Mamba-3) — too small for benefit"],
            "variants": [
                {
                    "name": "mamba",
                    "source": "scripts/models/mamba_hybrid.py (hand-written)",
                    "upgrades": [
                        "complex_valued_A (Mamba-3)",
                        "trapezoidal_discretization (Mamba-3)",
                        "bidirectional",
                    ],
                },
                {
                    "name": "mamba_ssm_v1",
                    "source": "mamba_ssm.Mamba (Tri Dao fused CUDA kernel)",
                    "upgrades": [
                        "real_diagonal_A (Mamba-1 reference)",
                        "euler_discretization (Mamba-1 reference)",
                        "bidirectional",
                    ],
                },
            ],
        },
        "feature_columns_used": list(feature_cols_used or []),
        "drop_columns": list(ID_COLS),
        "imputation": {
            "strategy": "column_median (global, pre-CV)",
            "fills_low_noise_csv": imputation_fills_first or {},
            "note": "inherited from raw extract_features.py NaN on 12 rows, same across noise levels",
        },
        "timestamp_utc": timestamp,
        "total_seconds": float(time.time() - run_started),
        "per_noise_total_seconds": {k: float(v) for k, v in per_noise_seconds.items()},
    }
    (RESULTS_DIR / f"synthetic_run_log{sfx}.json").write_text(
        json.dumps(run_log, indent=2), encoding="utf-8"
    )

    print("\nSummary (window-F1 / chunk-F1):")
    mean_rows = metrics_df[metrics_df["fold"] == "mean"].copy()
    for _, r in mean_rows.iterrows():
        print(f"  [{r['noise_level']:<4s}] {r['model']:<8s}  window={r['macro_f1']:.3f}  chunk={r['chunk_macro_f1']:.3f}")

    print(f"\nWrote metrics:    {RESULTS_DIR / 'synthetic_cv_metrics.csv'}")
    print(f"Wrote OOF preds:  {RESULTS_DIR / 'synthetic_oof_predictions.csv'}")
    print(f"Wrote importance: {RESULTS_DIR / 'synthetic_feature_importance.csv'}")
    print(f"Wrote CM json:    {RESULTS_DIR / 'synthetic_confusion_matrices.json'}")
    print(f"Wrote run log:    {RESULTS_DIR / 'synthetic_run_log.json'}")


if __name__ == "__main__":
    main()
