# GPU Server Runbook — Synthetic Texture Classification Experiment

This document hands off the remaining work to a GPU machine. The laptop side
has already finished data generation, model code, and CPU smoke tests. All
code is version-controlled on the branch `chore/complete-analytics-loop`.

The remaining work is:

1. Full training run (tabular + Mamba) on the GPU server.
2. Figure generation.
3. Result validation.
4. Phase D — conference-style report (writer/reviewer loop).
5. Phase E — Beamer slides (writer/reviewer loop).

---

## Status snapshot at handoff

| Phase | Status |
|---|---|
| A — research & style audit | ✅ done |
| B — synthetic data generation (static RECIPE + α sinusoid + β AR(2)) | ✅ done |
| C1 — trainer code (`scripts/train_synthetic.py`) | ✅ written, CPU-smoke-verified for 5/6 models |
| C2 — figures code (`scripts/make_synthetic_figures.py`) | ✅ written, not yet exercised |
| C — full 3 × 6 × 5-fold run | ⏳ blocked on compute (Mamba ≈ 21 h on laptop CPU) |
| D — conference-style report | ⏳ writer/reviewer loop, needs Phase C outputs |
| E — Beamer slides | ⏳ after Phase D |

### Known smoke F1s (fold-0, low noise, CPU)

| model | macro-F1 | fit time |
|---|---:|---:|
| logreg | 0.498 | 0.0 s |
| rf | 0.408 | 1.3 s |
| xgb | 0.403 | 3.2 s |
| mlp | 0.466 | 1.5 s |
| ftt | 0.438 | 35.0 s |
| mamba (3 ep) | 0.334 | 379 s |
| mamba (9 ep) | 0.392 (rising) | — |

Tabular models sit on a ~0.4–0.5 plateau because two of the 19 features
(`jaw_lr_delta_abs_mean`, `mouth_open_smooth_mean`) carry temporal-only
signal (α sinusoid + β AR(2)) that tabular models cannot decode. The
smoking-gun ablation (T5 tabular-only-2cols F1 = 0.319 vs T6 chunk-summary
F1 = 0.947) confirms this.

Expected with full Mamba + GPU: **window-F1 ≈ 0.80–0.90, chunk-F1 ≈ 0.90+**
on low noise, a clean F1-decay curve toward ~0.4 at high noise.

---

## Repo layout

```
INDENG 243/
├── GPU_SERVER_PLAN.md            <- this file
├── README.md
├── requirements.txt              <- pinned deps (pip)
├── texture_label_map.json        <- (unused for synthetic experiment)
├── configs/label_schema.yaml
├── data/                         <- raw per-video CSVs, gitignored
├── artifacts/
│   ├── features/
│   │   ├── window_level_features.csv                   <- 8077 × 25, PRE-existing
│   │   ├── window_level_features_synthetic_{low,med,high}.csv   <- Phase B output
│   │   └── synthetic_generation_params_{low,med,high}.json
│   └── results/                                        <- Phase C output lands here
├── reports/
│   ├── figures/                  <- Phase C output lands here
│   ├── *.md                      <- legacy analytics reports (unrelated to this experiment)
│   └── synthetic_texture_paper*.md  <- Phase D output (writer/reviewer loop)
└── scripts/
    ├── generate_synthetic_labels.py     <- Phase B generator
    ├── train_synthetic.py               <- Phase C trainer (CLI-configurable)
    ├── make_synthetic_figures.py        <- Phase C figures
    └── models/
        └── mamba_hybrid.py              <- hand-written Mamba block
```

Everything under `artifacts/features/window_level_features_synthetic_*.csv`
and `artifacts/features/synthetic_generation_params_*.json` is **already
generated** and committed, so the synthetic data is reproducible without
re-running the generator.

---

## 1. Environment setup on the GPU server

### 1.1 Python

Use Python 3.10+ (3.12 tested locally).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pulls in:
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`, `pyyaml`, `matplotlib`, `seaborn` (existing)
- `xgboost` (for the GBM baseline; needs `libomp` on macOS — not a concern on Linux)
- `torch` (CPU or CUDA wheel — your call)
- `rtdl-revisiting-models` (FT-Transformer by Yandex)

### 1.2 CUDA torch

If your server has CUDA, install the matching wheel:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

Adjust cu121 to whatever your driver wants. Then verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True NVIDIA ...
```

### 1.3 (macOS only, not server) libomp workaround

On macOS the CPU smoke tests needed `brew install libomp` and
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` to avoid
a `libomp.dylib` collision between torch and xgboost. Linux servers do
not have this problem; no env-var workaround needed.

---

## 2. Regenerate synthetic data (optional — already committed)

```bash
python scripts/generate_synthetic_labels.py
```

Expected stdout: 3 noise levels, 8077 windows, 129 chunks, balanced labels
(medium 2923, hard 2635, soft 2519), temporal sanity: soft β-acf≈0,
medium ≈0.3, hard ≈0.7; α period 3 / 7 / 14.

Outputs overwrite:
```
artifacts/features/window_level_features_synthetic_{low,med,high}.csv
artifacts/features/synthetic_generation_params_{low,med,high}.json
```

**If the CSVs are already present, skip this step** — the random seed is
fixed (42) so results are deterministic anyway.

---

## 3. Full training run (Phase C)

### 3.1 One command, everything on GPU

```bash
python scripts/train_synthetic.py \
  --device cuda \
  --models logreg,rf,xgb,mlp,ftt,mamba \
  --noise-levels low,med,high \
  --mamba-epochs 80 \
  --mamba-d-model 64 --mamba-n-blocks 2 --mamba-d-state 16 \
  --nn-epochs 60
```

### 3.2 Split run (if you want to reuse CPU results from laptop)

```bash
# on laptop (CPU) — already runnable but slow; tabular only
python scripts/train_synthetic.py \
  --device cpu --models logreg,rf,xgb,mlp,ftt \
  --noise-levels low,med,high \
  --results-suffix _cpu

# on GPU server — Mamba only
python scripts/train_synthetic.py \
  --device cuda --models mamba \
  --noise-levels low,med,high \
  --mamba-epochs 80 \
  --results-suffix _gpu

# then merge CSVs (simple pandas concat; trivial to write a helper)
```

### 3.3 Expected outputs

```
artifacts/results/
├── synthetic_cv_metrics.csv            <- per-fold + fold=mean rows, window/chunk F1
├── synthetic_oof_predictions.csv       <- per-window OOF y_pred + probabilities
├── synthetic_feature_importance.csv    <- logreg/rf/xgb only
├── synthetic_confusion_matrices.json
└── synthetic_run_log.json              <- hyperparams + timing
```

### 3.4 Expected timing (rough, single H100/A100)

| step | wall clock |
|---|---|
| tabular 5 models × 3 noise × 5 fold | ~5 min |
| FT-Transformer (40 ep × 3 × 5) | ~3 min |
| **Mamba hybrid** (80 ep × 3 × 5, bidirectional) | **~30 min** |
| Total | **~40 min** |

On CPU the same Mamba config would take ~21 hours — hence the GPU move.

### 3.5 Expected headline results

If everything is wired correctly:

```
noise=low
  logreg   window_F1 ≈ 0.55   chunk_F1 ≈ 0.65
  rf       window_F1 ≈ 0.58   chunk_F1 ≈ 0.70
  xgb      window_F1 ≈ 0.60   chunk_F1 ≈ 0.72
  mlp      window_F1 ≈ 0.57   chunk_F1 ≈ 0.68
  ftt      window_F1 ≈ 0.60   chunk_F1 ≈ 0.72
  mamba    window_F1 ≈ 0.82   chunk_F1 ≈ 0.92   <-- the breakthrough

noise=high
  all models collapse toward chance F1 ≈ 0.35 (decay curve)
```

Exact numbers will vary; directional behavior must hold or there is a bug.

---

## 4. Figures (Phase C)

```bash
python scripts/make_synthetic_figures.py
```

Produces, both `.png` (DPI 320) and `.svg`, under `reports/figures/`:

- `synthetic_01_f1_decay_curve`
- `synthetic_02_confusion_matrices_grid`
- `synthetic_03_feature_importance_low`
- `synthetic_04_per_class_f1_by_noise`
- `synthetic_05_probability_calibration`

---

## 5. Result validator (cross-check)

Spawn an **Explore agent** (if using Claude Code) with the prompt template
below. It must not see the trainer code — only the artifacts. Its job is
to independently recompute macro-F1 from `synthetic_oof_predictions.csv`,
verify per-class counts, and flag suspicious patterns.

> **Validator prompt skeleton (fill in paths):**
> Read `artifacts/results/synthetic_oof_predictions.csv`. For each
> `(noise_level, model)`:
> 1. Recompute window-level macro-F1 from `y_true`/`y_pred`.
> 2. Recompute per-chunk majority-vote macro-F1.
> 3. Check prob columns sum to ~1 per row.
> 4. Check that OOF video_ids cover all 20 videos exactly once.
> 5. Flag: Mamba F1 not > best tabular by ≥0.1 on low noise → suggests a bug.
> 6. Flag: high-noise Mamba F1 > 0.6 → suggests leakage.
>
> Cross-reference with `synthetic_cv_metrics.csv` (the trainer's own report)
> and report any discrepancy. Return a short PASS/FAIL audit.

---

## 6. Phase D — conference-style report (writer/reviewer loop)

Use TWO agents, alternating. Each round produces a file under
`reports/synthetic_texture_paper_r{N}.md` and a matching review under
`reports/synthetic_texture_paper_r{N}_review.md`.

### 6.1 Paper skeleton (ICLR/NeurIPS style)

```
Abstract (150–250 w)

1. Introduction
   - Problem: chewing-behavior video → food-texture classification
   - Challenge: labels sparse + inherently temporal
   - Contributions (4 bullets):
     * synthetic-data recipe isolating tabular-visible vs temporal-only signal
     * hand-written Mamba hybrid (Mamba-1 base + Mamba-3 complex-A + trapezoidal + bidirectional)
     * decay-curve protocol quantifying temporal contribution
     * smoking-gun ablation: T5 tabular F1=0.32 vs T6 chunk-summary F1=0.95

2. Related Work  (0.5 page)
   - Selective SSMs: Mamba-1 (Gu & Dao 2023), Mamba-2 (Dao & Gu 2024), Mamba-3 (ICLR 2026)
   - Tabular NNs: MLP, FT-Transformer (Gorishniy et al. 2021)
   - Chewing/landmark analysis: MediaPipe pipelines, prior texture classification work
   - Synthetic-signal benchmarks: precedents in controlled ablations

3. Method
   3.1 Dataset and window-level features (8077 windows, 20 videos, 19 features)
   3.2 Synthetic label injection
       - Chunk assignment (5–8 contiguous chunks per video, balanced labels)
       - Static RECIPE (tabular-visible magnitude shifts)
       - α sinusoidal signal (period 3/7/14 windows by label)
       - β AR(2) drift (roots 0/0.42/0.86 lag-1 acf by label)
       - Invariance argument: per-chunk std label-invariant by construction
   3.3 Models
       - Classical: LogReg / RF / XGBoost
       - Tabular NN: MLP (128-64-32) + FT-Transformer (2-block, d=64)
       - Sequence: bidirectional Mamba hybrid (Mamba-1 selective SSM +
         Mamba-3 complex-valued A (data-dependent rotary) +
         Mamba-3 trapezoidal discretization; MIMO SSM omitted for scale)
   3.4 Cross-validation protocol
       - StratifiedGroupKFold(5) on video_id
       - Window-F1 (8077 predictions) and chunk-F1 (majority vote, 129 chunks)

4. Experiments
   4.1 Main result: F1-decay curve across noise levels × models
   4.2 Per-class and per-chunk analysis (confusion matrices)
   4.3 Smoking-gun ablation: T5 vs T6
   4.4 (optional) Mamba-3 upgrade ablation: complex-A off / trapezoidal → Euler

5. Discussion & Limitations
   - Synthetic-label scaffolding vs real deployment
   - Scale: 20 videos is an experimental unit count, not a statistical one
   - Generalization risk, chunking artifacts, etc.

6. Conclusion

References
Appendix A: hyperparameters, full per-class tables
Appendix B: training curves (per-fold loss + F1)
```

### 6.2 Writer / Reviewer loop

| Round | Writer agent | Reviewer agent |
|---|---|---|
| R1 | Reads `synthetic_cv_metrics.csv`, `synthetic_run_log.json`, `synthetic_generation_params_*.json`, `reports/figures/*`; drafts full paper | Acts as ICLR area-chair-level reviewer; 20-bullet critique, rates 6/10 |
| R2 | Addresses each reviewer bullet with rebuttal notes inline (in a comment block) + revises | Focuses on residual issues + rates 7/10 |
| R3 *(optional)* | Final polish: section-level coherence, figure-caption clarity, notation consistency | Copy-edit pass |

R3's writer output is the final deliverable (`reports/synthetic_texture_paper.md`).

### 6.3 Reviewer prompt hints

- Challenge every causal claim ("Mamba learns temporal structure" → is the ablation actually cleanly showing this?)
- Require that every number in the body is traceable to a file in `artifacts/results/`
- Expect explicit limitations (20 videos, synthetic labels, chunking artifact)
- Expect a Mamba-3 ablation table even if small (complex-A on/off, trapezoidal vs Euler)
- Expect a "why not Mamba-2 SSD" one-paragraph justification

---

## 7. Phase E — Beamer slides

Use `metropolis` theme (or `Copenhagen` as a fallback if `metropolis.sty`
is unavailable). Writer/reviewer loop, same agent pattern as Phase D.

### 7.1 Slide outline (14 slides, ~15-min talk)

| # | Slide | Source |
|---|---|---|
| 1 | Title | (author + date) |
| 2 | Motivation | Introduction of the paper |
| 3 | Setup & data | § 3.1 |
| 4 | Static signal (RECIPE) | § 3.2 |
| 5 | Temporal signals (α sin + β AR(2)) | § 3.2 — critical slide, visual of the two waveforms per label |
| 6 | Model zoo | § 3.3 |
| 7 | Mamba hybrid block diagram | § 3.3 — TikZ figure recommended |
| 8 | CV protocol + metrics | § 3.4 |
| 9 | Main result — F1-decay curve | Figure synthetic_01 |
| 10 | Confusion matrices + chunk F1 | Figures synthetic_02 + per-chunk bar |
| 11 | Smoking-gun T5 vs T6 | § 4.3 |
| 12 | Mamba-3 ablation | § 4.4 |
| 13 | Limitations | § 5 |
| 14 | Conclusion + Q&A | § 6 |

### 7.2 Compilation

```bash
latexmk -pdf -outdir=reports/slides reports/slides/synthetic_texture_talk.tex
```

---

## 8. Gotchas observed during development

1. **libomp conflict on macOS**: torch + xgboost share `libomp.dylib`.
   Fix: install brew libomp AND pass `OMP_NUM_THREADS=1
   MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` as env vars, AND import
   xgboost before torch. Linux/server avoids this.

2. **12 rows of NaN features in the base CSV**: inherited from the
   upstream `extract_features.py` pipeline. `train_synthetic.py` median-
   imputes these rows globally before CV. If you want fold-local
   imputation, swap the logic in `main()` — but 12/8077 rows is noise.

3. **`person_id` is all NaN** in the base CSV, so CV falls back to
   `video_id` as the group column. StratifiedGroupKFold is used when
   possible; falls back to GroupKFold otherwise. Recorded in
   `synthetic_run_log.json → cv_strategy`.

4. **Mamba complex-A numerical stability**: `A_real_log` is the log of
   positive values and `A_real = -exp(·)` stays negative (stable). `A_imag`
   is unconstrained — initialized to `N(0, 0.1)`. Large imaginary parts
   can oscillate; monitor training loss for NaN.

5. **Sequential scan cost**: the Python for-loop over timesteps is
   ~126 s/epoch on CPU for the full config. On GPU it should be
   ~3-5 s/epoch. If you still see slow behavior, consider:
   - `torch.compile(clf)` before the training loop
   - Reduce `d_state` (16 → 8) for ~2× speedup

6. **Deterministic seeds**: generator seed = 42, StratifiedGroupKFold
   random_state = 42, torch manual_seed = 42 + fold. Full pipeline is
   reproducible.

---

## 9. Quick verification checklist before reporting

- [ ] `artifacts/results/synthetic_cv_metrics.csv` exists and has rows for
      every `(noise, model, fold)` + `fold=mean` rows (≈ 18 × 6 / 108 + 18
      aggregate = ~126 rows).
- [ ] Mamba fold=mean window-F1 at low noise > 0.7.
- [ ] Mamba fold=mean window-F1 at high noise < 0.55 (decay visible).
- [ ] Tabular fold=mean window-F1 at low noise in [0.45, 0.65].
- [ ] `synthetic_run_log.json → mamba_config.upgrades` lists
      `complex_valued_A` and `trapezoidal_discretization`.
- [ ] All five figures exist as both `.png` and `.svg` under
      `reports/figures/`.
- [ ] Result-validator agent signs off PASS.
- [ ] Phase D R3 paper compiles (markdown only, no LaTeX build needed here).
- [ ] Phase E Beamer PDF compiles.

---

## 10. Open questions / decisions deferred to the GPU run

1. **Mamba-3 ablation**: whether to run an explicit complex-A-off and
   trapezoidal→Euler toggle. Recommended: yes, as a small appendix table
   (2 extra runs on low-noise only, ~6 min extra).
2. **Chunk-level metric**: paper reports both window- and chunk-level F1.
   Decide whether to headline one or both.
3. **Noise axis**: currently 3 discrete levels (0.1 / 0.3 / 0.6 σ). If
   time permits, add 0.05 and 0.9 for a smoother decay curve (2 more
   noise settings = 2 more generator + trainer invocations).

All of these are ≤ 30 min of extra compute; decide at report-draft time.

---

## Contact / handoff notes

- Branch: `chore/complete-analytics-loop`.
- Base repo: `/Users/prophecia/Desktop/INDENG 243`.
- Working laptop tree has committed everything in the repo layout above.
- Raw videos (`*.mov`) and `data/` are gitignored — **do not need to
  re-extract features**; the committed `artifacts/features/...` is enough.
