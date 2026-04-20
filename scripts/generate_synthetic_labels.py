"""Generate synthetic window-level food_type labels for the synthetic texture
classification experiment.

Pipeline per video:
    1. sort windows chronologically (by window_id)
    2. split the sequence into N contiguous chunks (N drawn from [5, 8])
    3. assign each chunk a food_type (soft / medium / hard), balanced per video
    4. apply the STATIC per-label perturbation recipe (multiplicative / additive)
    5. apply the TEMPORAL-ONLY signals (α sinusoidal + β AR(2) drift) per chunk
       — designed to be invisible to tabular aggregates but recoverable by
       sequence models (Mamba, RNN, etc.)
    6. add Gaussian noise scaled by noise_level * feature_sigma_original
    7. clip the result to the original p5/p95 bounds

Outputs (per noise level):
    artifacts/features/window_level_features_synthetic_<level>.csv
    artifacts/features/synthetic_generation_params_<level>.json

The original window_level_features.csv is never mutated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_IN = PROJECT_ROOT / "artifacts" / "features" / "window_level_features.csv"
OUT_DIR = PROJECT_ROOT / "artifacts" / "features"

FOOD_TYPES = ["soft", "medium", "hard"]

# ---------------------------------------------------------------------------
# STATIC per-label perturbation recipe (window-aggregate magnitudes).
# mode=mul -> value *= factor; mode=add -> value += factor (absolute units).
# Medium is intentionally identity so it acts as the baseline class.
# These signals are VISIBLE to tabular models via feature means.
# ---------------------------------------------------------------------------
RECIPE = {
    "mouth_open_smooth_mean": {"soft": 0.90, "medium": 1.00, "hard": 1.15, "mode": "mul"},
    "mouth_open_smooth_std":  {"soft": 0.70, "medium": 1.00, "hard": 1.40, "mode": "mul"},
    "mouth_open_smooth_p95":  {"soft": 0.90, "medium": 1.00, "hard": 1.15, "mode": "mul"},
    "mouth_open_px_mean":     {"soft": 0.90, "medium": 1.00, "hard": 1.15, "mode": "mul"},
    "mouth_open_px_std":      {"soft": 0.70, "medium": 1.00, "hard": 1.40, "mode": "mul"},
    "mouth_open_px_p95":      {"soft": 0.90, "medium": 1.00, "hard": 1.15, "mode": "mul"},
    "jaw_lr_delta_abs_mean":  {"soft": 0.85, "medium": 1.00, "hard": 1.30, "mode": "mul"},
    "pause_ratio":            {"soft": -0.05, "medium": 0.00, "hard": 0.08, "mode": "add"},
}

# ---------------------------------------------------------------------------
# TEMPORAL-ONLY signals applied per chunk. Design philosophy:
#   We REPLACE the within-chunk fluctuations of two target features with a
#   label-specific pattern, preserving the per-chunk mean (so the static RECIPE
#   signal carried by chunk means is intact) and using a label-invariant
#   per-chunk target std (so tabular models cannot exploit variance).
#   Only the within-chunk WAVEFORM varies by label — only sequence models can
#   decode that.
# ---------------------------------------------------------------------------

# α : sinusoidal waveform with label-specific period (windows per cycle).
# Replaces within-chunk fluctuations of ALPHA_TARGET_FEATURE with sin(·)
# scaled to the global median within-chunk std of that feature.
ALPHA_TARGET_FEATURE = "jaw_lr_delta_abs_mean"
ALPHA_PERIOD_PER_LABEL = {"soft": 3, "medium": 7, "hard": 14}

# β : AR(2) drift with label-specific coefficients.
# Replaces within-chunk fluctuations of BETA_TARGET_FEATURE with drift scaled
# to the global median within-chunk std of that feature.
# Stability: (0,0) white; (0.5,-0.2) damped osc; (1.2,-0.4) persistent peak.
# Expected lag-1 autocorr: soft≈0, medium≈0.42, hard≈0.86 (Yule-Walker).
BETA_TARGET_FEATURE = "mouth_open_smooth_mean"
BETA_AR2_COEFS = {
    "soft":   (0.00,  0.00),
    "medium": (0.50, -0.20),
    "hard":   (1.20, -0.40),
}

# Gaussian observation noise added on top of the pattern (fraction of target std).
TEMPORAL_OBS_NOISE_FRAC = 0.20

NOISE_LEVELS = {"low": 0.10, "med": 0.30, "high": 0.60}

N_CHUNKS_MIN = 5
N_CHUNKS_MAX = 8
RANDOM_SEED = 42


def split_into_chunks(n_windows: int, n_chunks: int) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for n_chunks contiguous near-equal slices."""
    n_chunks = min(n_chunks, n_windows)
    sizes = [n_windows // n_chunks] * n_chunks
    for i in range(n_windows % n_chunks):
        sizes[i] += 1
    bounds, start = [], 0
    for s in sizes:
        bounds.append((start, start + s))
        start += s
    return bounds


def assign_labels_per_video(n_chunks: int, rng: np.random.Generator) -> list[str]:
    """Assign n_chunks labels with per-video balance (equal when divisible by 3)."""
    base = n_chunks // 3
    rem = n_chunks - base * 3
    labels = ["soft"] * base + ["medium"] * base + ["hard"] * base
    labels.extend(rng.choice(FOOD_TYPES, size=rem, replace=True).tolist())
    rng.shuffle(labels)
    return labels


def alpha_signal(n: int, label: str, rng: np.random.Generator) -> np.ndarray:
    """Sinusoid normalized to ~unit std. Period varies by label."""
    period = ALPHA_PERIOD_PER_LABEL[label]
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    t = np.arange(n, dtype=float)
    raw = np.sin(2.0 * np.pi * t / period + phase)
    s = raw.std()
    if s > 1e-12:
        raw = raw / s
    return raw


def beta_signal(n: int, label: str, rng: np.random.Generator) -> np.ndarray:
    """AR(2) drift normalized to unit std per chunk. Coefs vary by label."""
    rho1, rho2 = BETA_AR2_COEFS[label]
    eps = rng.normal(0.0, 1.0, size=n)
    drift = np.zeros(n, dtype=float)
    if n >= 1:
        drift[0] = eps[0]
    if n >= 2:
        drift[1] = rho1 * drift[0] + eps[1]
    for t in range(2, n):
        drift[t] = rho1 * drift[t - 1] + rho2 * drift[t - 2] + eps[t]
    s = drift.std()
    if s > 1e-12:
        drift = drift / s
    return drift


def _chunk_acf_lag1(values: np.ndarray) -> float:
    v = values - np.mean(values)
    var = np.var(v)
    if var < 1e-12 or len(v) < 2:
        return float("nan")
    return float(np.mean(v[:-1] * v[1:]) / var)


def _chunk_dominant_period(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    n = len(centered)
    if n < 4 or np.var(centered) < 1e-12:
        return float("nan")
    spec = np.abs(np.fft.rfft(centered))
    if len(spec) <= 1:
        return float("nan")
    k = int(np.argmax(spec[1:]) + 1)
    return float(n / k)


def generate(level_name: str, noise_scale: float, random_seed: int) -> dict:
    rng = np.random.default_rng(random_seed)
    df = pd.read_csv(FEATURES_IN)
    df = df.sort_values(["video_id", "window_id"]).reset_index(drop=True)

    clip_bounds: dict[str, tuple[float, float]] = {}
    feature_sigma: dict[str, float] = {}
    for col in RECIPE:
        series = df[col].dropna()
        lo, hi = np.percentile(series, [5, 95])
        clip_bounds[col] = (float(lo), float(hi))
        feature_sigma[col] = float(series.std())

    # --- Step 1: assign chunks and labels per video ---
    df["food_type_synthetic"] = ""
    df["chunk_id"] = ""
    chunk_label_map: dict[str, str] = {}
    chunk_index_map: dict[str, list[int]] = {}
    global_chunk_counter = 0
    for video_id, group in df.groupby("video_id", sort=True):
        n_win = len(group)
        n_chunks = int(rng.integers(N_CHUNKS_MIN, N_CHUNKS_MAX + 1))
        bounds = split_into_chunks(n_win, n_chunks)
        labels = assign_labels_per_video(len(bounds), rng)
        group_idx = group.index.tolist()
        for (s, e), label in zip(bounds, labels):
            chunk_name = f"v{int(video_id)}_c{global_chunk_counter:04d}"
            idx_slice = group_idx[s:e]
            df.loc[idx_slice, "food_type_synthetic"] = label
            df.loc[idx_slice, "chunk_id"] = chunk_name
            chunk_label_map[chunk_name] = label
            chunk_index_map[chunk_name] = idx_slice
            global_chunk_counter += 1
    total_chunks = global_chunk_counter

    # --- Step 2: static per-label perturbations (vectorized by label mask) ---
    for col, spec in RECIPE.items():
        original = df[col].to_numpy(dtype=float)
        perturbed = original.copy()
        label_col = df["food_type_synthetic"].to_numpy()
        for label in FOOD_TYPES:
            mask = label_col == label
            if not mask.any():
                continue
            if spec["mode"] == "mul":
                perturbed[mask] = original[mask] * spec[label]
            else:
                perturbed[mask] = original[mask] + spec[label]
        df[col] = perturbed

    # --- Step 3: REPLACE within-chunk content with label-specific temporal pattern.
    # We keep per-chunk MEAN (which carries the static RECIPE signal) and set
    # per-chunk STD to a GLOBAL median within-chunk std (label-invariant) so
    # tabular models cannot decode label from variance — only sequence models
    # can from the waveform structure.

    def _global_median_within_chunk_std(col: str) -> float:
        stds = []
        for idx_list in chunk_index_map.values():
            if len(idx_list) >= 2:
                stds.append(df.loc[idx_list, col].std())
        return float(np.median(stds)) if stds else float(df[col].std())

    alpha_target_std = _global_median_within_chunk_std(ALPHA_TARGET_FEATURE)
    beta_target_std = _global_median_within_chunk_std(BETA_TARGET_FEATURE)

    for chunk_name, idx_list in chunk_index_map.items():
        label = chunk_label_map[chunk_name]
        n = len(idx_list)

        # α target: sinusoid + obs noise, scaled to target std, centered on chunk mean
        base_mean_a = float(df.loc[idx_list, ALPHA_TARGET_FEATURE].mean())
        a_pat = alpha_signal(n, label, rng)
        a_noise = rng.normal(0.0, TEMPORAL_OBS_NOISE_FRAC, size=n)
        df.loc[idx_list, ALPHA_TARGET_FEATURE] = base_mean_a + (a_pat + a_noise) * alpha_target_std

        # β target: AR(2) drift + obs noise, scaled to target std, centered on chunk mean
        base_mean_b = float(df.loc[idx_list, BETA_TARGET_FEATURE].mean())
        b_pat = beta_signal(n, label, rng)
        b_noise = rng.normal(0.0, TEMPORAL_OBS_NOISE_FRAC, size=n)
        df.loc[idx_list, BETA_TARGET_FEATURE] = base_mean_b + (b_pat + b_noise) * beta_target_std

    # --- Step 4: Gaussian noise and clip to original p5/p95 ---
    for col in RECIPE:
        sigma = feature_sigma[col]
        perturbed = df[col].to_numpy(dtype=float)
        noise = rng.normal(0.0, noise_scale * sigma, size=perturbed.shape[0])
        df[col] = np.clip(perturbed + noise, clip_bounds[col][0], clip_bounds[col][1])

    # --- Save CSV and params ---
    out_csv = OUT_DIR / f"window_level_features_synthetic_{level_name}.csv"
    params_path = OUT_DIR / f"synthetic_generation_params_{level_name}.json"
    df.to_csv(out_csv, index=False)

    params = {
        "level_name": level_name,
        "noise_scale_sigma_multiplier": noise_scale,
        "random_seed": random_seed,
        "n_windows": int(len(df)),
        "n_videos": int(df["video_id"].nunique()),
        "n_chunks_total": int(total_chunks),
        "chunks_per_video_range": [N_CHUNKS_MIN, N_CHUNKS_MAX],
        "features_perturbed_static": list(RECIPE.keys()),
        "recipe_static": RECIPE,
        "temporal_alpha": {
            "target_feature": ALPHA_TARGET_FEATURE,
            "mode": "REPLACE within-chunk fluctuations with unit-std sinusoid, scaled to alpha_target_std",
            "period_per_label": ALPHA_PERIOD_PER_LABEL,
            "phase": "uniform(0, 2π) per chunk",
            "obs_noise_fraction_of_target_std": TEMPORAL_OBS_NOISE_FRAC,
            "alpha_target_std": float(alpha_target_std),
        },
        "temporal_beta": {
            "target_feature": BETA_TARGET_FEATURE,
            "mode": "REPLACE within-chunk fluctuations with unit-std AR(2) drift, scaled to beta_target_std",
            "ar2_coefficients_per_label": {k: list(v) for k, v in BETA_AR2_COEFS.items()},
            "obs_noise_fraction_of_target_std": TEMPORAL_OBS_NOISE_FRAC,
            "beta_target_std": float(beta_target_std),
        },
        "feature_clip_bounds_p5_p95": {k: list(v) for k, v in clip_bounds.items()},
        "feature_sigma_original": feature_sigma,
        "label_distribution": {k: int(v) for k, v in df["food_type_synthetic"].value_counts().items()},
        "input_csv": str(FEATURES_IN.relative_to(PROJECT_ROOT)),
        "output_csv": str(out_csv.relative_to(PROJECT_ROOT)),
    }
    with params_path.open("w") as f:
        json.dump(params, f, indent=2)

    # --- Sanity printouts ---
    print(f"[{level_name}] noise={noise_scale}  windows={len(df)}  chunks={total_chunks}")
    print(f"[{level_name}] label dist: {params['label_distribution']}")
    print(f"[{level_name}] STATIC per-label feature means:")
    for col in RECIPE:
        means = df.groupby("food_type_synthetic")[col].mean().round(3).to_dict()
        print(f"  {col:<30s} {means}")

    print(f"[{level_name}] TEMPORAL sanity (per-chunk measures averaged by label):")
    per_label_acf: dict[str, list[float]] = {lbl: [] for lbl in FOOD_TYPES}
    per_label_period: dict[str, list[float]] = {lbl: [] for lbl in FOOD_TYPES}
    per_label_chunk_std_alpha: dict[str, list[float]] = {lbl: [] for lbl in FOOD_TYPES}
    per_label_chunk_std_beta: dict[str, list[float]] = {lbl: [] for lbl in FOOD_TYPES}
    for chunk_name, idx_list in chunk_index_map.items():
        lbl = chunk_label_map[chunk_name]
        b_vals = df.loc[idx_list, BETA_TARGET_FEATURE].to_numpy(dtype=float)
        a_vals = df.loc[idx_list, ALPHA_TARGET_FEATURE].to_numpy(dtype=float)
        per_label_acf[lbl].append(_chunk_acf_lag1(b_vals))
        per_label_period[lbl].append(_chunk_dominant_period(a_vals))
        per_label_chunk_std_alpha[lbl].append(float(np.std(a_vals)))
        per_label_chunk_std_beta[lbl].append(float(np.std(b_vals)))

    for lbl in FOOD_TYPES:
        acfs = np.array([a for a in per_label_acf[lbl] if not np.isnan(a)])
        pers = np.array([p for p in per_label_period[lbl] if not np.isnan(p)])
        a_stds = np.array([s for s in per_label_chunk_std_alpha[lbl] if not np.isnan(s)])
        b_stds = np.array([s for s in per_label_chunk_std_beta[lbl] if not np.isnan(s)])
        expected_period = ALPHA_PERIOD_PER_LABEL[lbl]
        print(
            f"  [{lbl:<6s}] "
            f"β acf(lag1)={acfs.mean() if len(acfs) else float('nan'):+.3f} (expect soft≈0 med≈0.4 hard≈0.85)  "
            f"α period={pers.mean() if len(pers) else float('nan'):.2f} (expect ≈{expected_period})  "
            f"α chunk-std={a_stds.mean() if len(a_stds) else float('nan'):.2f}  "
            f"β chunk-std={b_stds.mean() if len(b_stds) else float('nan'):.2f}"
        )
    print(f"[{level_name}] wrote {out_csv.name} + {params_path.name}\n")
    return params


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for level_name, noise_scale in NOISE_LEVELS.items():
        generate(level_name, noise_scale, RANDOM_SEED)


if __name__ == "__main__":
    main()
