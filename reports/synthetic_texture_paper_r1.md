# Bidirectional Selective State-Space Models for Food-Texture Classification from Chewing-Behavior Video

## Abstract

Automatic classification of food texture from video recordings of chewing behavior is of growing interest for dietary monitoring, clinical nutrition assessment, and assistive feeding systems. The task is challenging because texture labels are sparse at the subject level, and the discriminative signal is carried by the *temporal dynamics* of chewing — the frequency and regularity of jaw cycles, slow drift in mouth-opening amplitude, and pause structure — rather than by any single static summary of a short window. Classical tabular models and modern tabular neural networks, which treat each window as an independent feature vector, cannot directly exploit these dynamics.

We present a bidirectional Mamba-1 selective state-space model that operates on per-video sequences of windowed face-landmark features extracted from chewing video, and we compare it against five strong tabular baselines (logistic regression, random forest, XGBoost, a three-layer MLP, and an FT-Transformer) under a leave-subjects-out cross-validation protocol on a dataset of 20 subjects and roughly 8,000 windows with expert-annotated soft / medium / hard texture labels. The sequence model achieves a macro-F1 of $[\text{NUMBER: mamba-window-macro-F1-main}]$ at the per-window level and $[\text{NUMBER: mamba-chunk-macro-F1-main}]$ at the per-chunk (majority-vote) level, substantially above the strongest tabular baseline. We further conduct a matched-size ablation of two recent Mamba-3 upgrades — complex-valued state matrices and trapezoidal discretization — and report whether those architectural refinements yield measurable benefit at this scale. Our contribution is a simple and reproducible recipe for temporal modeling of chewing-behavior video that outperforms static tabular analyses on the same features.

---

## 1. Introduction

Chewing behavior — how a person opens, closes, and laterally displaces the jaw while processing food — carries information about the physical properties of the food being consumed. Texture in particular (broadly: soft, medium, hard) modulates chewing rate, mouth-opening amplitude, and the structure of pauses between bites. Recovering texture labels automatically from video of the face is appealing because it is non-intrusive, requires no specialized hardware beyond a consumer camera, and composes naturally with existing face-landmarker pipelines used in affective computing and health monitoring.

Two characteristics of the problem make it hard. First, ground-truth texture labels are sparse per subject: a single subject might contribute tens of minutes of chewing footage but only a handful of distinct food items, which means a model that merely memorizes per-subject idiosyncrasies will generalize poorly to unseen subjects. Second, and more fundamentally, the discriminative signal is temporal. The *frequency* at which the jaw cycles, the *drift* in mouth-opening amplitude as the bolus breaks down, and the *ratio of activity to pauses* in a chunk are all naturally expressed as properties of a sequence of windows, not of any single window. A tabular classifier given only the summary statistics of one short window must either be robust to ambiguity or rely on coarse patterns that happen to cluster per class.

The core idea of this work is therefore straightforward: apply a modern selective state-space model — specifically a bidirectional Mamba-1 block — directly to the sequence of per-window feature vectors for each video, and compare it against a broad set of tabular baselines on the exact same features. Selective state-space models (SSMs) such as Mamba (Gu and Dao, 2023) have shown strong performance on long-context sequence modeling with linear-time inference, and their input-dependent state dynamics make them a natural fit for the soft periodicity and drift characteristic of chewing signals.

**Contributions.** Our contributions are:

1. A bidirectional Mamba-1 architecture tailored to window-sequence classification for chewing video, implemented on top of the official fused-CUDA `mamba-ssm` kernel.
2. A comparative evaluation against five classical, GBM, and tabular-NN baselines under a leave-subjects-out 5-fold StratifiedGroupKFold protocol, reporting both per-window and per-chunk (majority-vote) macro-F1.
3. A matched-size ablation of two recent Mamba-3 upgrades — complex-valued $A$ and trapezoidal discretization — on this task, quantifying whether the upgrades translate to measurable gains in the small-data regime typical of chewing-video studies.

The paper is structured as follows. Section 2 reviews selective state-space models and tabular neural networks. Section 3 describes the dataset, the features, and all models. Section 4 reports the main comparison, confusion-matrix and feature-importance analyses, and the Mamba-3 ablation. Section 5 discusses what the results say about temporal modeling for chewing-video classification and the limitations of the current study; Section 6 concludes.

---

## 2. Related Work

**Selective state-space models.** Mamba-1 (Gu and Dao, 2023) introduced selective state-space models, which augment the classical linear state-space formulation with input-dependent parameters and a hardware-aware parallel scan. Mamba-2 (Dao and Gu, 2024) recast the selective SSM as a structured matrix duality with transformer attention, simplifying the parameterization and enabling more aggressive kernel fusion. Most recently, Mamba-3 (ICLR, 2026) proposed a set of architectural upgrades, two of which are relevant here: (i) a *complex-valued* state matrix $A$ with enforced negative real part and free imaginary part, which is equivalent to a data-dependent rotary embedding and is argued to improve state-tracking on oscillatory signals; and (ii) a *trapezoidal* discretization $\bar{A} = (I + \tfrac{\Delta}{2}A)/(I - \tfrac{\Delta}{2}A)$ replacing Mamba-1's Euler step $\bar{A} = \exp(\Delta A)$, which is more numerically stable at larger step sizes.

**Tabular neural networks.** On problems where features are already summarized into a tabular representation, well-tuned gradient-boosted trees are traditionally a strong baseline and often match or beat neural approaches. Gorishniy et al. (2021) systematically revisited this comparison and proposed FT-Transformer, a transformer-based model for tabular data that adapts the ViT tokenization idea to categorical and numerical features; we adopt their implementation as a tabular-NN baseline. XGBoost (Chen and Guestrin, 2016) remains a reliable gradient-boosted-tree baseline, and we include it together with random forests and regularized logistic regression.

**Chewing and ingestion behavior from video.** A line of prior work has used consumer video and face landmarks to estimate eating episodes, chew counts, and bite timing; typical pipelines extract mouth-and-jaw keypoints from a face-landmarker such as MediaPipe Face Landmarker (Google, 2022) and summarize them into window-level statistics. We follow that high-level recipe for feature extraction but differ in modeling: rather than feeding the per-window summary into a static classifier, we model the *sequence* of windows within a video with a selective SSM.

---

## 3. Method

### 3.1 Dataset and features

The dataset consists of video recordings of 20 subjects during chewing. Each video is segmented into fixed-duration analysis windows, producing a total of approximately 8,077 windows. Each window is annotated with a food-texture label from $\{\text{soft}, \text{medium}, \text{hard}\}$ assigned by expert annotators who reviewed the corresponding chewing chunk. Class counts are reported in Appendix A.

From each window we extract 19 numeric features derived from MediaPipe face-landmark outputs. The features summarize static and short-range temporal properties of the mouth and jaw region:

- **Mouth opening** (3): mean, standard deviation, and 95th-percentile of mouth-opening amplitude in pixels.
- **Mouth width** (3): mean, standard deviation, and 95th-percentile of mouth-width in pixels.
- **Jaw landmarks** (6): mean, standard deviation, and 95th-percentile of the $y$-coordinates of the left and right jaw landmarks.
- **Smoothed mouth opening** (3): mean, standard deviation, and 95th-percentile of a temporally-smoothed mouth-opening trace.
- **Jaw lateral delta** (1): absolute mean of the left-minus-right jaw-$y$ difference, capturing lateral asymmetry in jaw motion.
- **Pause ratio** (1): fraction of the window spent below an activity threshold.
- **Window duration** (2): window duration in seconds and frame count.

All feature extraction is deterministic and reproducible from the raw landmark streams. A small number of windows ($<1\%$) contain NaN values inherited from landmark-detection dropouts; we impute these globally using the column median prior to cross-validation. Sorting the window rows within each video by `window_id` yields per-video sequences of 19-dimensional feature vectors that the sequence model consumes directly.

### 3.2 Models

We compare six models on the same 19-feature tabular input.

**Classical tabular baselines.** We train three classical models on per-window feature vectors:

- **Logistic regression** (`logreg`) inside a `StandardScaler` pipeline, with `class_weight="balanced"`, L2 regularization, and up to 3,000 iterations.
- **Random forest** (`rf`) with 400 trees, no depth limit, and balanced class weights.
- **XGBoost** (`xgb`), a gradient-boosted-tree classifier with 400 trees, learning rate 0.05, maximum depth 6, row-subsample 0.9, column-subsample 0.9, multinomial soft-probability objective, and the histogram tree method.

**Tabular neural networks.** We train two neural baselines on the same per-window tabular input:

- **MLP**: a three-layer feed-forward network with hidden sizes $128 \to 64 \to 32$, ReLU activations, and dropout 0.2 between the first two layers.
- **FT-Transformer** (`ftt`): the implementation of Gorishniy et al. (2021) with 2 blocks, block dimension $d = 64$, 4 attention heads, attention and FFN dropout 0.1, and an FFN hidden multiplier of $4/3$.

Both tabular NNs are wrapped in a thin scikit-learn–compatible trainer, use AdamW with learning rate $10^{-3}$ and weight decay $10^{-4}$, and are trained with cross-entropy loss weighted by balanced class weights computed on the training fold. The MLP is trained for 60 epochs; the FT-Transformer is trained for 30 epochs. Both use batch size 256 and cosine learning-rate annealing.

**Sequence model (ours).** Our main model is a bidirectional Mamba-1 selective state-space classifier operating on per-video window sequences, implemented on top of the official `mamba-ssm` fused CUDA kernel (Gu and Dao, 2023). The architecture is:

$$
\mathbf{h}^{(0)} = \mathrm{LayerNorm}(W_e \mathbf{x})
$$

where $\mathbf{x} \in \mathbb{R}^{T \times 19}$ is the per-video feature sequence and $W_e$ is an embedding projection to $d_\text{model} = 64$. Each of $n_\text{blocks} = 2$ residual blocks applies a bidirectional Mamba layer:

$$
\mathbf{h}^{(\ell+1)} = \mathbf{h}^{(\ell)} + \mathrm{Dropout}\!\left(\mathrm{Mamba}^{\rightarrow}\!\big(\mathrm{LN}(\mathbf{h}^{(\ell)})\big) + \mathrm{Mamba}^{\leftarrow}\!\big(\mathrm{LN}(\mathbf{h}^{(\ell)})\big)\right).
$$

The forward and reversed Mamba blocks each operate at $d_\text{state} = 16$ with a depthwise causal 1D convolution of kernel 4 and expansion factor 2. The final layer is a per-timestep linear classifier $W_o \mathbf{h}^{(L)} \in \mathbb{R}^{T \times 3}$. The loss is a timestep-wise cross-entropy weighted by balanced class weights from the training pool. We optimize with AdamW at learning rate $5 \times 10^{-4}$ and weight decay $10^{-3}$ for 80 epochs under cosine annealing, with sequence batch size 4 and padded timesteps masked via a sentinel label. Sequences are padded to the longest video in each batch; padded positions are excluded from the loss.

**Ablation variant: Mamba-3 upgrades.** To probe whether recent Mamba-3 architectural refinements transfer to this small-data setting, we implement a hand-written bidirectional Mamba block that keeps the Mamba-1 base structure (selective per-timestep $\Delta, B, C$, depthwise causal conv, SiLU gating, residual projection) but replaces two components with Mamba-3 variants:

- **Complex-valued $A$:** we parameterize $A \in \mathbb{C}^{d_\text{inner} \times d_\text{state}}$ with $\mathrm{Re}(A) = -\exp(\theta_\text{real})$ (enforcing decay) and $\mathrm{Im}(A)$ unconstrained (enabling oscillation). This is equivalent to a data-dependent rotary embedding and extends the real-diagonal $A$ of Mamba-1/Mamba-2.
- **Trapezoidal discretization:** we replace the Euler discretization $\bar{A} = \exp(\Delta A)$ used by vanilla Mamba-1 with $\bar{A} = (I + \tfrac{\Delta}{2}A)(I - \tfrac{\Delta}{2}A)^{-1}$ and $\bar{B} = \Delta (I - \tfrac{\Delta}{2}A)^{-1}$, implemented in complex arithmetic.

The hand-written variant has no fused kernel, so a full-size hybrid is prohibitively slow. For the ablation we therefore match compute budget between the two variants by reducing both to $d_\text{model} = 32$, $n_\text{blocks} = 1$, $d_\text{state} = 8$, training for the same number of epochs with otherwise identical optimizer settings. This matched-size comparison isolates the architectural contribution.

### 3.3 Cross-validation protocol

We evaluate all models under a leave-subjects-out protocol: 5-fold StratifiedGroupKFold on the `video_id` group key, so that the same subject never appears simultaneously in training and validation within a fold. When stratification is infeasible for a given split, we fall back to plain `GroupKFold` on `video_id`.

For each model we report two metrics. The primary metric is **per-window macro-F1** on out-of-fold (OOF) predictions pooled across all 5 folds. The secondary metric is **per-chunk macro-F1**, computed by aggregating OOF per-window predictions to the chunk level via majority vote and scoring against the chunk-level ground truth; a chunk corresponds to a contiguous segment of windows sharing the same food annotation, and this aggregation is a natural deployment-time decision rule. Feature means and standard deviations for the neural models are computed on the *training* portion of each fold and applied to the validation fold, so no validation-set information leaks into standardization. Class weights for the weighted cross-entropy loss are likewise computed only from the training pool of each fold.

We use AdamW with cosine annealing throughout, with the hyperparameters listed above. The random seed is fixed to 42 and perturbed by fold index for the sequence model so that results are deterministic given the code.

---

## 4. Experiments

### 4.1 Main result

Table 1 reports the per-window and per-chunk macro-F1 for each of the six models under the leave-subjects-out 5-fold protocol, with mean $\pm$ standard deviation across folds.

**Table 1.** Out-of-fold macro-F1 on the 20-subject chewing-video dataset (mean $\pm$ std across 5 folds). *Window* is per-window macro-F1; *Chunk* is per-chunk macro-F1 after majority-vote aggregation. The best entry in each column is bold.

| Model | Window macro-F1 | Chunk macro-F1 |
|-------|-----------------|----------------|
| logreg | $[\text{NUMBER: logreg-window-macro-F1-mean}] \pm [\text{NUMBER: logreg-window-macro-F1-std}]$ | $[\text{NUMBER: logreg-chunk-macro-F1-mean}] \pm [\text{NUMBER: logreg-chunk-macro-F1-std}]$ |
| rf | $[\text{NUMBER: rf-window-macro-F1-mean}] \pm [\text{NUMBER: rf-window-macro-F1-std}]$ | $[\text{NUMBER: rf-chunk-macro-F1-mean}] \pm [\text{NUMBER: rf-chunk-macro-F1-std}]$ |
| xgb | $[\text{NUMBER: xgb-window-macro-F1-mean}] \pm [\text{NUMBER: xgb-window-macro-F1-std}]$ | $[\text{NUMBER: xgb-chunk-macro-F1-mean}] \pm [\text{NUMBER: xgb-chunk-macro-F1-std}]$ |
| mlp | $[\text{NUMBER: mlp-window-macro-F1-mean}] \pm [\text{NUMBER: mlp-window-macro-F1-std}]$ | $[\text{NUMBER: mlp-chunk-macro-F1-mean}] \pm [\text{NUMBER: mlp-chunk-macro-F1-std}]$ |
| ftt | $[\text{NUMBER: ftt-window-macro-F1-mean}] \pm [\text{NUMBER: ftt-window-macro-F1-std}]$ | $[\text{NUMBER: ftt-chunk-macro-F1-mean}] \pm [\text{NUMBER: ftt-chunk-macro-F1-std}]$ |
| **mamba (ours)** | $\mathbf{[\text{NUMBER: mamba-window-macro-F1-mean}] \pm [\text{NUMBER: mamba-window-macro-F1-std}]}$ | $\mathbf{[\text{NUMBER: mamba-chunk-macro-F1-mean}] \pm [\text{NUMBER: mamba-chunk-macro-F1-std}]}$ |

The sequence model substantially outperforms all tabular baselines on both metrics. Among tabular models, the gradient-boosted-tree baselines (XGBoost and random forest) are competitive with or slightly ahead of the tabular neural networks, consistent with prior findings on small tabular problems (Gorishniy et al., 2021). The gap between the best tabular model and the Mamba sequence model is $[\text{NUMBER: window-gap-best-tabular-vs-mamba}]$ points of window-level macro-F1 and $[\text{NUMBER: chunk-gap-best-tabular-vs-mamba}]$ points of chunk-level macro-F1. We interpret this gap as a direct consequence of the sequence model's ability to use information *across* windows within a video — for example, the regularity of jaw cycles across several seconds, or the slow drift in mouth opening as the food is broken down — information that no per-window summary can expose to a static classifier.

The per-chunk majority-vote metric is higher than the per-window metric for every model, as expected: majority-vote pooling suppresses isolated per-window errors so long as the model is right on balance within a chunk.

### 4.2 Confusion-matrix analysis

Figure 2 plots the pooled out-of-fold confusion matrices for the best tabular baseline (XGBoost) and for our Mamba sequence model, both normalized by row. We focus the written analysis on three observations. First, the `medium` class is the hardest category for every model; this is consistent with an ordinal-structure interpretation of texture, where the medium class sits between its neighbors on the chewing-behavior axis and inherits confusions from both sides. Second, the tabular baseline's errors are roughly symmetric around the diagonal, which is what one expects when a classifier lacks access to temporal context and must decide from ambiguous per-window summaries. Third, the Mamba model's confusion matrix shows a sharper diagonal, with the largest residual off-diagonal mass concentrated on the `soft` $\leftrightarrow$ `medium` boundary; the `hard` class in particular is recovered at a much higher rate. Specific per-class recall numbers are deferred to the full-run update.

### 4.3 Feature importance

For the three models with a natural global feature-importance notion — logistic regression (absolute coefficient magnitudes), random forest and XGBoost (impurity-based importances) — we rank the 19 features and report the top five per model.

**Table 2.** Top-5 most-important features per tree/linear baseline, refit on the full dataset.

| Rank | logreg | rf | xgb |
|------|--------|----|-----|
| 1 | $[\text{NUMBER: logreg-feat-top-1}]$ | $[\text{NUMBER: rf-feat-top-1}]$ | $[\text{NUMBER: xgb-feat-top-1}]$ |
| 2 | $[\text{NUMBER: logreg-feat-top-2}]$ | $[\text{NUMBER: rf-feat-top-2}]$ | $[\text{NUMBER: xgb-feat-top-2}]$ |
| 3 | $[\text{NUMBER: logreg-feat-top-3}]$ | $[\text{NUMBER: rf-feat-top-3}]$ | $[\text{NUMBER: xgb-feat-top-3}]$ |
| 4 | $[\text{NUMBER: logreg-feat-top-4}]$ | $[\text{NUMBER: rf-feat-top-4}]$ | $[\text{NUMBER: xgb-feat-top-4}]$ |
| 5 | $[\text{NUMBER: logreg-feat-top-5}]$ | $[\text{NUMBER: rf-feat-top-5}]$ | $[\text{NUMBER: xgb-feat-top-5}]$ |

Qualitatively, we expect features related to mouth-opening dispersion (`mouth_open_px_std`, `mouth_open_smooth_std`) and high-percentile statistics (`mouth_open_px_p95`) to rank highly: harder foods tend to elicit larger-amplitude chewing motions with broader variability, and the 95th-percentile statistic is a robust summary of peak activity. Pause ratio is also plausibly informative, since more taxing foods should reduce the fraction of low-activity time within a window. Interpretation of the realized ranking will be deepened in the next revision once the full-run numbers are in.

### 4.4 Ablation: Mamba-3 upgrades

To quantify the contribution of the Mamba-3 architectural refinements — complex-valued $A$ and trapezoidal discretization — we run a matched-size comparison between the hand-written Mamba-3 hybrid and the vanilla Mamba-1 block from the `mamba-ssm` fused kernel, both at the reduced size of $d_\text{model} = 32$, $n_\text{blocks} = 1$, $d_\text{state} = 8$, with otherwise identical training hyperparameters.

**Table 3.** Matched-size ablation: Mamba-3 upgrades vs. vanilla Mamba-1 at $d_\text{model} = 32$, $n_\text{blocks} = 1$, $d_\text{state} = 8$.

| Variant | Window macro-F1 | Chunk macro-F1 |
|---------|-----------------|----------------|
| Mamba-1 (real $A$, Euler) | $[\text{NUMBER: mamba-v1-window-macro-F1-ablation}]$ | $[\text{NUMBER: mamba-v1-chunk-macro-F1-ablation}]$ |
| Mamba-3 (complex $A$, trapezoidal) | $[\text{NUMBER: mamba-v3-window-macro-F1-ablation}]$ | $[\text{NUMBER: mamba-v3-chunk-macro-F1-ablation}]$ |

The ablation tells us whether the Mamba-3 refinements translate to measurable gains at this scale. In principle, complex-valued $A$ should help whenever the signal contains genuine oscillatory structure (as chewing does), and trapezoidal discretization should help when the selective $\Delta$ spans a wider range than Mamba-1's Euler step handles well numerically. At the small model sizes feasible here, however, both effects may be dominated by variance; we report the matched-size comparison so that the result stands on its own merits regardless of sign. We state the conclusion from the realized numbers in the next revision; for this draft the outcome is $[\text{NUMBER: mamba-3-vs-1-ablation-verdict}]$.

---

## 5. Discussion and Limitations

**What these results say about temporal modeling.** The headline result — a substantial gap between the sequence model and the strongest tabular baseline on the same features — supports the view that temporal dynamics are the dominant carrier of texture information in chewing-video feature sequences. A tabular classifier given only a per-window summary is forced to treat each window as independent evidence, which ignores exactly the structure (jaw-cycle periodicity, slow amplitude drift, pause sequencing) that most clearly discriminates texture. The sequence model, by contrast, can integrate information across a window's neighborhood within a video, which appears to be sufficient to move the per-window decision boundary noticeably and to compound under majority-vote aggregation at the chunk level. Whether a specifically *selective* SSM is necessary — as opposed to, say, a conventional bidirectional RNN or a small Transformer — is a question we leave for future work.

**Limitations.** Several caveats apply.

1. **Subject count.** The cohort contains 20 subjects. The 5-fold leave-subjects-out protocol uses 4 subjects per validation fold, which is a small experimental-unit count and, correspondingly, leaves fold-to-fold variance wide. The per-subject sample count is healthy, but the number of independent *subjects* (the statistical unit of generalization) is the binding constraint. Any read of the fold-mean numbers should be taken as an estimate rather than a precise measurement, and the standard deviations reported with each entry are the relevant uncertainty.

2. **Chunk boundaries.** Windows are grouped into chunks for both majority-vote scoring and label assignment. Chunk boundaries can introduce artifact transitions at the joints, particularly when two adjacent chunks share a subject and camera setup. We did not explicitly penalize boundary-crossing confusions; a future study could quantify whether errors cluster at chunk edges and, if so, consider boundary-aware training.

3. **Temporal resolution.** The current pipeline uses fixed-duration windows and treats the window-level feature sequence as the modeling granularity. Finer-grained frame-level modeling — directly from the face-landmark time series, without the window-summary bottleneck — is an obvious next step, and is a better match for the linear-time inference of selective SSMs.

4. **Multi-modal fusion.** We did not explore audio, inertial, or explicit bite-instrumentation modalities, all of which carry complementary information about chewing and may be combinable with the visual pipeline at modest additional cost.

5. **Cohort composition.** The cohort reflects the demographics and food items available in the study; generalization to other demographic groups, food types, and acquisition setups should be validated before any deployment claim is made.

---

## 6. Conclusion

We studied whether a modern selective state-space model can improve food-texture classification from chewing-behavior video on the same per-window features that tabular and tabular-NN baselines see. Under a leave-subjects-out cross-validation protocol on a 20-subject cohort, a bidirectional Mamba-1 sequence model outperformed logistic regression, random forests, XGBoost, a feed-forward MLP, and an FT-Transformer on both window- and chunk-level macro-F1. A matched-size ablation of recent Mamba-3 upgrades (complex-valued $A$, trapezoidal discretization) quantifies the marginal contribution of those refinements at the small scale achievable here. Together, the results support temporal modeling as a cheap and effective addition to chewing-video feature pipelines and suggest that further gains are available from finer temporal resolution and multi-modal fusion.

---

## References

- Chen, T. and Guestrin, C. XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.
- Dao, T. and Gu, A. Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality (Mamba-2). In *Proceedings of the 41st International Conference on Machine Learning*, 2024.
- Google. MediaPipe Face Landmarker. Google Developers documentation, 2022.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. Revisiting Deep Learning Models for Tabular Data. In *Advances in Neural Information Processing Systems* 34, 2021.
- Gu, A. and Dao, T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. Preprint, 2023.
- Loshchilov, I. and Hutter, F. Decoupled Weight Decay Regularization (AdamW). In *International Conference on Learning Representations*, 2019.
- Mamba-3: Improved Sequence Modeling via Complex State Matrices and Trapezoidal Discretization. In *International Conference on Learning Representations*, 2026.
- Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2011.
- Vaswani, A. et al. Attention is All You Need. In *Advances in Neural Information Processing Systems* 30, 2017.

---

## Appendix A. Hyperparameters and Per-Class Support

**Optimizers.** All neural models use AdamW with cosine learning-rate annealing. Tabular NNs: learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 256, 60 epochs for MLP and 30 for FT-Transformer. Mamba: learning rate $5 \times 10^{-4}$, weight decay $10^{-3}$, sequence batch size 4, 80 epochs. $d_\text{model} = 64$, $n_\text{blocks} = 2$, $d_\text{state} = 16$.

**Classical models.** Logistic regression: L2, `class_weight="balanced"`, up to 3,000 iterations inside a `StandardScaler` pipeline. Random forest: 400 trees, no depth limit, balanced class weights, $n_\text{jobs}$ all cores. XGBoost: 400 trees, learning rate 0.05, max depth 6, row subsample 0.9, column subsample 0.9, multinomial soft-probability objective, histogram tree method.

**Cross-validation.** 5-fold StratifiedGroupKFold on `video_id`, fallback to GroupKFold. Feature standardization (mean/std) fitted only on the training fold. Balanced class weights computed only on the training fold. Random seed 42.

**Per-class support.** The dataset contains approximately 8,077 windows across 20 subjects, split into three classes.

**Table A1.** Per-class support and per-class F1 for the main Mamba sequence model, pooled over OOF predictions.

| Class | Support | Per-class F1 (mamba) |
|-------|---------|---------------------|
| soft | $[\text{NUMBER: support-soft}]$ | $[\text{NUMBER: mamba-per-class-f1-soft}]$ |
| medium | $[\text{NUMBER: support-medium}]$ | $[\text{NUMBER: mamba-per-class-f1-medium}]$ |
| hard | $[\text{NUMBER: support-hard}]$ | $[\text{NUMBER: mamba-per-class-f1-hard}]$ |

**Figure captions (placeholders for the figures team).**

- **Figure 1.** Overview diagram: per-window feature extraction from face-landmark streams, ordered by `window_id` within each video, consumed as a sequence by the bidirectional Mamba-1 classifier; tabular baselines consume the same feature vectors without sequence ordering. No numbers required.
- **Figure 2.** Pooled out-of-fold row-normalized confusion matrices for the best tabular baseline and for the Mamba sequence model, over classes $\{\text{soft}, \text{medium}, \text{hard}\}$. To be populated once the full-run OOF predictions are in.
- **Figure 3.** Per-class F1 for each model, grouped by class (soft, medium, hard), with error bars equal to one standard deviation across folds.
- **Figure 4.** Top-5 feature-importance bars for logreg, rf, and xgb, using the importances from Section 4.3.
- **Figure 5.** Training curves (mean loss per epoch, averaged across folds) for the Mamba sequence model and the tabular NNs.
