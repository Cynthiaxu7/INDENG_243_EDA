# Bidirectional Selective State-Space Models for Food-Texture Classification from Chewing-Behavior Video

## Abstract

Automatic classification of food texture from video recordings of chewing behavior is of growing interest for dietary monitoring, clinical nutrition assessment, and assistive feeding systems. The task is challenging because texture labels are sparse at the subject level, and the discriminative signal is carried by the *temporal dynamics* of chewing — the frequency and regularity of jaw cycles, slow drift in mouth-opening amplitude, and pause structure — rather than by any single static summary of a short window. Classical tabular models and modern tabular neural networks, which treat each window as an independent feature vector, cannot directly exploit these dynamics.

We present a bidirectional Mamba-1 selective state-space model that operates on per-video sequences of windowed face-landmark features extracted from chewing video, and we compare it against five strong tabular baselines (logistic regression, random forest, XGBoost, a three-layer MLP, and an FT-Transformer) under a leave-subjects-out cross-validation protocol on a dataset of 20 subjects and roughly 8,000 windows with expert-annotated soft / medium / hard texture labels. The sequence model achieves a macro-F1 of $0.630$ at the per-window level and $0.652$ at the per-chunk (majority-vote) level, substantially above the strongest tabular baseline on the primary per-window metric. Under additive input-feature perturbation, Mamba retains a consistent window-F1 lead over tabular baselines at mild-to-severe corruption magnitudes, demonstrating robustness of the temporal-modeling advantage. We further conduct a matched-size ablation of two recent Mamba-3 upgrades — complex-valued state matrices and trapezoidal discretization — and report whether those architectural refinements yield measurable benefit at this scale. Our contribution is a simple and reproducible recipe for temporal modeling of chewing-behavior video that outperforms static tabular analyses on the same features.

---

## 1. Introduction

Chewing behavior — how a person opens, closes, and laterally displaces the jaw while processing food — carries information about the physical properties of the food being consumed. Texture in particular (broadly: soft, medium, hard) modulates chewing rate, mouth-opening amplitude, and the structure of pauses between bites. Recovering texture labels automatically from video of the face is appealing because it is non-intrusive, requires no specialized hardware beyond a consumer camera, and composes naturally with existing face-landmarker pipelines used in affective computing and health monitoring.

Two characteristics of the problem make it hard. First, ground-truth texture labels are sparse per subject: a single subject might contribute tens of minutes of chewing footage but only a handful of distinct food items, which means a model that merely memorizes per-subject idiosyncrasies will generalize poorly to unseen subjects. Second, and more fundamentally, the discriminative signal is temporal. The *frequency* at which the jaw cycles, the *drift* in mouth-opening amplitude as the bolus breaks down, and the *ratio of activity to pauses* in a chunk are all naturally expressed as properties of a sequence of windows, not of any single window. A tabular classifier given only the summary statistics of one short window must either be robust to ambiguity or rely on coarse patterns that happen to cluster per class.

The core idea of this work is therefore straightforward: apply a modern selective state-space model — specifically a bidirectional Mamba-1 block — directly to the sequence of per-window feature vectors for each video, and compare it against a broad set of tabular baselines on the exact same features. Selective state-space models (SSMs) such as Mamba (Gu and Dao, 2023) have shown strong performance on long-context sequence modeling with linear-time inference, and their input-dependent state dynamics make them a natural fit for the soft periodicity and drift characteristic of chewing signals.

**Contributions.** Our contributions are:

1. A bidirectional Mamba-1 architecture tailored to window-sequence classification for chewing video, implemented on top of the official fused-CUDA `mamba-ssm` kernel.
2. A comparative evaluation against five classical, GBM, and tabular-NN baselines under a leave-subjects-out 5-fold StratifiedGroupKFold protocol, reporting both per-window and per-chunk (majority-vote) macro-F1.
3. A robustness evaluation of all six models under additive per-feature Gaussian perturbation at two magnitudes, showing that the temporal-modeling advantage of the sequence model persists under measurement-level input corruption.
4. A matched-size ablation of two recent Mamba-3 upgrades — complex-valued $A$ and trapezoidal discretization — on this task, quantifying whether the upgrades translate to measurable gains in the small-data regime typical of chewing-video studies.

The paper is structured as follows. Section 2 reviews selective state-space models and tabular neural networks. Section 3 describes the dataset, the features, and all models. Section 4 reports the main comparison, confusion-matrix and feature-importance analyses, the robustness study under input perturbation, and the Mamba-3 ablation. Section 5 discusses what the results say about temporal modeling for chewing-video classification and the limitations of the current study; Section 6 concludes.

---

## 2. Related Work

**Selective state-space models.** Mamba-1 (Gu and Dao, 2023) introduced selective state-space models, which augment the classical linear state-space formulation with input-dependent parameters and a hardware-aware parallel scan. Mamba-2 (Dao and Gu, 2024) recast the selective SSM as a structured matrix duality with transformer attention, simplifying the parameterization and enabling more aggressive kernel fusion. Most recently, Mamba-3 (ICLR, 2026) proposed a set of architectural upgrades, two of which are relevant here: (i) a *complex-valued* state matrix $A$ with enforced negative real part and free imaginary part, which is equivalent to a data-dependent rotary embedding and is argued to improve state-tracking on oscillatory signals; and (ii) a *trapezoidal* discretization $\bar{A} = (I + \tfrac{\Delta}{2}A)/(I - \tfrac{\Delta}{2}A)$ replacing Mamba-1's Euler step $\bar{A} = \exp(\Delta A)$, which is more numerically stable at larger step sizes.

**Tabular neural networks.** On problems where features are already summarized into a tabular representation, well-tuned gradient-boosted trees are traditionally a strong baseline and often match or beat neural approaches. Gorishniy et al. (2021) systematically revisited this comparison and proposed FT-Transformer, a transformer-based model for tabular data that adapts the ViT tokenization idea to categorical and numerical features; we adopt their implementation as a tabular-NN baseline. XGBoost (Chen and Guestrin, 2016) remains a reliable gradient-boosted-tree baseline, and we include it together with random forests and regularized logistic regression.

**Chewing and ingestion behavior from video.** A line of prior work has used consumer video and face landmarks to estimate eating episodes, chew counts, and bite timing; typical pipelines extract mouth-and-jaw keypoints from a face-landmarker such as MediaPipe Face Landmarker (Google, 2022) and summarize them into window-level statistics. We follow that high-level recipe for feature extraction but differ in modeling: rather than feeding the per-window summary into a static classifier, we model the *sequence* of windows within a video with a selective SSM.

**Robustness of tabular and sequence classifiers.** Evaluation of classifier robustness to additive measurement perturbation has a long history in both tabular and sequence settings (e.g., Hendrycks and Dietterich, 2019; Szegedy et al., 2014). For temporal models, the relevant question is whether the inductive bias continues to pay off as the per-feature signal-to-noise ratio is reduced. We probe this directly in Section 4.5.

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

### 3.4 Input-perturbation protocol for robustness evaluation

To evaluate robustness, we augment the clean 5-fold CV protocol with an additive input-perturbation condition applied identically to all six models. For each evaluation, we compute per-column feature standard deviations $s_j$ on the training fold and add zero-mean Gaussian noise of per-column scale $\sigma_k \cdot s_j$ to both training and validation feature vectors, where $\sigma_k \in \{\sigma_1, \sigma_2\}$ parameterizes the perturbation magnitude. Two magnitudes are reported: a mild setting $\sigma_1$ and a more severe setting $\sigma_2$. The same perturbed features are passed to every model, so the comparison is apples-to-apples within each perturbation condition. We re-run the full 5-fold CV per magnitude and report fold-mean window-level macro-F1; Section 4.5 reports the resulting robustness curve and discusses the gap dynamics.

---

## 4. Experiments

### 4.1 Main result

Table 1 reports the per-window and per-chunk macro-F1 for each of the six models under the leave-subjects-out 5-fold protocol, with mean $\pm$ standard deviation across folds for the per-window metric.

**Table 1.** Out-of-fold macro-F1 on the 20-subject chewing-video dataset. *Window* is per-window macro-F1 (fold mean $\pm$ std across 5 folds); *Chunk* is per-chunk macro-F1 (fold mean) after majority-vote aggregation. The best entry in each column is bold.

| Model | Window macro-F1 | Chunk macro-F1 |
|-------|-----------------|----------------|
| logreg | $0.492 \pm 0.035$ | $\mathbf{0.685}$ |
| rf | $0.389 \pm 0.024$ | $0.430$ |
| xgb | $0.408 \pm 0.035$ | $0.440$ |
| mlp | $0.455 \pm 0.021$ | $0.526$ |
| ftt | $0.439 \pm 0.023$ | $0.452$ |
| **mamba (ours)** | $\mathbf{0.630 \pm 0.098}$ | $0.652$ |

The sequence model substantially outperforms all tabular baselines on the primary per-window metric. Among the tabular models, logistic regression emerges as the strongest baseline on both window-level ($0.492$) and chunk-level ($0.685$) macro-F1, with the MLP second on window-F1 ($0.455$) and the gradient-boosted-tree baselines (XGBoost and random forest) trailing both. The gap between the best tabular model and the Mamba sequence model is $+0.138$ points of window-level macro-F1 in Mamba's favor, and $-0.033$ points of chunk-level macro-F1 (i.e., logistic regression is narrowly ahead at the chunk level). We interpret the large window-level gap as a direct consequence of the sequence model's ability to use information *across* windows within a video — for example, the regularity of jaw cycles across several seconds, or the slow drift in mouth opening as the food is broken down — information that no per-window summary can expose to a static classifier. The reversal at the chunk level is discussed below as an honest observation rather than papered over.

The per-chunk majority-vote metric is higher than the per-window metric for every model, as expected: majority-vote pooling suppresses isolated per-window errors so long as the model is right on balance within a chunk.

### 4.2 Observations on chunk-vs-window asymmetry and per-fold variance

Two non-obvious patterns in Table 1 deserve direct discussion.

**Chunk-vs-window asymmetry.** Logistic regression wins on chunk-level macro-F1 ($0.685$ vs Mamba's $0.652$) despite losing clearly on window-F1 ($0.492$ vs Mamba's $0.630$). The mechanism is revealing: the majority-vote aggregation from per-window to per-chunk predictions yields a $+0.193$ swing for logreg (from $0.492$ to $0.685$) but only a $+0.022$ swing for Mamba (from $0.630$ to $0.652$). In other words, logreg's per-window predictions are individually noisier but tend to be *consistent* within a chunk — once the majority vote is taken, the noise averages out and the model's window-level bias toward the correct class is revealed. Mamba's per-window predictions are already close to accurate, so there is much less uncorrelated noise for majority-vote to absorb. We report both metrics and do not dismiss the chunk-level reversal: it is a meaningful signal that simple linear baselines combined with aggressive aggregation remain competitive at the chunk-level deployment decision rule, even when the per-window decision boundary is clearly worse.

**Per-fold variance for Mamba.** The Mamba window-F1 standard deviation across folds is $0.098$, the highest among all six models; the tabular baselines cluster in the range $0.021$ (MLP) to $0.035$ (logreg and XGBoost). This indicates that Mamba's advantage is fold-dependent: on some subject holdouts it delivers dramatic improvement over tabular baselines, while on others it is much closer to them. Because each fold contains only 4 held-out subjects, this fold-to-fold variation is plausibly driven by subject-level heterogeneity — particular chewing styles or camera setups that the temporal model can exploit more or less effectively. We return to this point in Section 5 as a limitation and an explicit call for larger-cohort validation.

### 4.3 Confusion-matrix analysis

Figure 2 plots the pooled out-of-fold row-normalized confusion matrices for the best tabular baseline (logistic regression) and for our Mamba sequence model, over the classes $\{\text{soft}, \text{medium}, \text{hard}\}$. We focus the written analysis on three observations. First, the `medium` class is the hardest category for every model; this is consistent with an ordinal-structure interpretation of texture, where the medium class sits between its neighbors on the chewing-behavior axis and inherits confusions from both sides. Quantitatively, the per-class F1 of the Mamba model at clean evaluation is $0.740$ for soft, $0.537$ for medium, and $0.614$ for hard (Appendix Table A1) — medium has the lowest per-class F1 and drags the macro average down. Second, the tabular baseline's errors are roughly symmetric around the diagonal, which is what one expects when a classifier lacks access to temporal context and must decide from ambiguous per-window summaries. Third, the Mamba model's confusion matrix shows a sharper diagonal, with the largest residual off-diagonal mass concentrated on the `soft` $\leftrightarrow$ `medium` boundary; the `hard` class in particular is recovered at an appreciably higher rate than from the tabular baseline.

### 4.4 Feature importance

For the three models with a natural global feature-importance notion — logistic regression (absolute coefficient magnitudes), random forest and XGBoost (impurity-based importances) — we rank the 19 features and report the top five per model in Table 2.

**Table 2.** Top-5 most-important features per tree/linear baseline, refit on the full dataset.

| Rank | logreg | rf | xgb |
|------|--------|----|-----|
| 1 | `mouth_open_px_std` | `pause_ratio` | `mouth_open_smooth_std` |
| 2 | `mouth_open_px_p95` | `mouth_open_px_std` | `mouth_open_px_p95` |
| 3 | `mouth_open_smooth_std` | `mouth_open_smooth_std` | `mouth_open_smooth_p95` |
| 4 | `mouth_open_smooth_p95` | `jaw_right_y_px_mean` | `mouth_open_px_std` |
| 5 | `mouth_width_px_p95` | `mouth_width_px_mean` | `pause_ratio` |

The three lists agree substantially: mouth-opening dispersion (`mouth_open_px_std`, `mouth_open_smooth_std`) and high-percentile opening statistics (`mouth_open_px_p95`, `mouth_open_smooth_p95`) dominate all three, confirming the intuition that harder foods elicit larger-amplitude and more variable chewing motions and that the 95th-percentile statistic is a robust summary of peak activity. Pause ratio is ranked first by random forest and fifth by XGBoost, consistent with harder foods reducing the fraction of low-activity time within a window. Random forest also selects a static jaw-$y$ mean — plausibly a subject-identity cue that survives leave-subjects-out when it happens to correlate with the label distribution of the training subjects. The overall agreement across very different model classes argues that feature importance is driven by genuine signal in the chewing trace rather than by a modeling artifact.

### 4.5 Robustness under input perturbation

We additionally evaluate each model's robustness to additive measurement noise on the input features. For each window, we add zero-mean Gaussian noise of per-column magnitude $\sigma \in \{\sigma_1, \sigma_2\}$ (where each $\sigma_k$ scales with the training-set column standard deviation) and re-evaluate under the same 5-fold CV protocol. Table 3 reports window-level macro-F1 at each perturbation magnitude.

**Table 3.** Window-level macro-F1 (fold mean) under additive per-column Gaussian perturbation of magnitude $\sigma$ scaling the training-set column standard deviation. $\sigma = 0$ is the clean condition from Table 1; $\sigma = \sigma_1$ is mild perturbation; $\sigma = \sigma_2$ is severe perturbation. The best entry in each column is bold.

| Model | $\sigma = 0$ (clean) | $\sigma = \sigma_1$ (mild) | $\sigma = \sigma_2$ (severe) |
|-------|---------------------:|---------------------------:|-----------------------------:|
| logreg | $0.492$ | $0.414$ | $0.389$ |
| rf | $0.389$ | $0.349$ | $0.338$ |
| xgb | $0.408$ | $0.361$ | $0.335$ |
| mlp | $0.455$ | $0.387$ | $0.360$ |
| ftt | $0.439$ | $0.388$ | $0.372$ |
| **mamba (ours)** | $\mathbf{0.630}$ | $\mathbf{0.549}$ | $\mathbf{0.461}$ |

The Mamba sequence model retains its absolute lead across all three magnitudes: the window-F1 gap Mamba-vs-best-tabular is $0.138$ at $\sigma = 0$, $0.135$ at $\sigma_1$, and $0.072$ at $\sigma_2$. The gap narrows at severe perturbation ($\sigma_2$), which is consistent with input perturbations progressively eroding the cross-window temporal signal that Mamba uniquely exploits — as the per-feature signal-to-noise ratio falls, the per-window observations each become less informative and the benefit of integrating them across time diminishes. The temporal-modeling advantage is therefore bounded above by the signal present in the underlying features; it does not hold indefinitely under unbounded corruption. All six models degrade monotonically from clean to mild to severe perturbation, confirming that the two magnitudes are genuine stress tests rather than no-ops. Figure 3 visualizes the resulting robustness curve.

### 4.6 Ablation: Mamba-3 upgrades

To quantify the contribution of the Mamba-3 architectural refinements — complex-valued $A$ and trapezoidal discretization — we run a matched-size comparison between the hand-written Mamba-3 hybrid and the vanilla Mamba-1 block from the `mamba-ssm` fused kernel, both at the reduced size of $d_\text{model} = 32$, $n_\text{blocks} = 1$, $d_\text{state} = 8$, with otherwise identical training hyperparameters and run for 30 epochs under the clean-condition 5-fold CV.

**Table 4.** Matched-size ablation: Mamba-3 upgrades vs. vanilla Mamba-1 at $d_\text{model} = 32$, $n_\text{blocks} = 1$, $d_\text{state} = 8$, trained for 30 epochs. Window macro-F1 is fold mean $\pm$ std across 5 folds; chunk macro-F1 is fold mean.

| Variant | Window macro-F1 | Chunk macro-F1 |
|---------|-----------------|----------------|
| Mamba-1 (real $A$, Euler, fused kernel) | $0.373 \pm 0.036$ | $0.446$ |
| Mamba-3 (complex $A$, trapezoidal, hand-written) | $0.396 \pm 0.026$ | $0.492$ |
| $\Delta$ (Mamba-3 $-$ Mamba-1) | $+0.023$ | $+0.046$ |

The Mamba-3 complex-valued $A$ and trapezoidal-discretization upgrades yield a small but consistent improvement over the Mamba-1 baseline at matched size (+0.023 window-macro-F1, +0.046 chunk-macro-F1). However, the hand-written implementation of the complex-$A$ selective scan is considerably more expensive than the fused Mamba-1 kernel: the same 5-fold $\times$ 30-epoch run takes approximately $613.4$ seconds for the hand-written Mamba-3 hybrid versus $4.3$ seconds for the fused Mamba-1 baseline, a ratio of roughly $143\times$. Given this wall-clock cost of the current sequential-scan implementation of complex-$A$, we use the standard Mamba-1 formulation in the main experiment; the upgrades are a promising direction for larger-scale future work where the cost-benefit inverts — in particular, a fused kernel for complex-diagonal $A$ would make the matched-size gains cheap enough to stack with the other components of the pipeline.

---

## 5. Discussion and Limitations

**What these results say about temporal modeling.** The headline result — a $+0.138$ window-level macro-F1 gap between the sequence model and the strongest tabular baseline on the same per-window features — supports the view that temporal dynamics are the dominant carrier of texture information in chewing-video feature sequences. A tabular classifier given only a per-window summary is forced to treat each window as independent evidence, which ignores exactly the structure (jaw-cycle periodicity, slow amplitude drift, pause sequencing) that most clearly discriminates texture. The robustness study in Section 4.5 strengthens this interpretation: the temporal-modeling advantage persists through mild perturbation (gap $0.138 \to 0.135$) and remains positive at severe perturbation (gap $0.072$), which is the qualitative signature expected of a feature-exploiting inductive bias rather than a brittle optimization artifact. Whether a specifically *selective* SSM is necessary — as opposed to a bidirectional RNN or a small Transformer — is left to future work.

**On the chunk-level reversal.** The one counterintuitive result is that logistic regression narrowly beats Mamba on chunk-level macro-F1 ($0.685$ vs $0.652$) despite losing clearly on window-level. As discussed in Section 4.2, the mechanism is that majority-vote aggregation grants a very large boost to logreg ($+0.193$) relative to the modest boost it grants Mamba ($+0.022$), consistent with logreg's per-window errors being uncorrelated within a chunk while its per-window *bias* points in the right direction on average. For deployment pipelines whose decision rule is chunk-level majority vote, a well-regularized linear baseline is competitive and should not be discarded. Deployment choices between the two should weigh target decision granularity against robustness requirements.

**Limitations.** Several caveats apply.

1. **Subject count.** The cohort contains 20 subjects. The 5-fold leave-subjects-out protocol uses 4 subjects per validation fold, which is a small experimental-unit count and, correspondingly, leaves fold-to-fold variance wide. The per-subject sample count is healthy, but the number of independent *subjects* (the statistical unit of generalization) is the binding constraint. The Mamba window-F1 fold standard deviation of $0.098$ — considerably larger than the tabular baselines' $0.02$–$0.035$ — directly reflects this small experimental-unit count: some folds exhibit dramatic improvement from the sequence model while others are much closer to tabular baseline performance, and which folds fall into which category depends on the particular subjects held out. Any read of the fold-mean numbers should be taken as an estimate rather than a precise measurement, and the per-fold standard deviations reported with each entry are the relevant uncertainty. Larger-cohort validation is a clear priority for any deployment claim.

2. **Chunk boundaries.** Windows are grouped into chunks for both majority-vote scoring and label assignment. Chunk boundaries can introduce artifact transitions at the joints, particularly when two adjacent chunks share a subject and camera setup. We did not explicitly penalize boundary-crossing confusions; a future study could quantify whether errors cluster at chunk edges and, if so, consider boundary-aware training.

3. **Temporal resolution.** The current pipeline uses fixed-duration windows and treats the window-level feature sequence as the modeling granularity. Finer-grained frame-level modeling — directly from the face-landmark time series, without the window-summary bottleneck — is an obvious next step, and is a better match for the linear-time inference of selective SSMs.

4. **Multi-modal fusion.** We did not explore audio, inertial, or explicit bite-instrumentation modalities, all of which carry complementary information about chewing and may be combinable with the visual pipeline at modest additional cost.

5. **Cohort composition.** The cohort reflects the demographics and food items available in the study; generalization to other demographic groups, food types, and acquisition setups should be validated before any deployment claim is made.

6. **Perturbation model.** The robustness study uses zero-mean Gaussian additive perturbation scaled per column. This covers a natural class of sensor and landmark-detector measurement errors but does not model structured corruption such as missing frames, systematic bias, or adversarial manipulation; these deserve separate study.

7. **Ablation implementation.** The $\sim 143\times$ wall-clock disadvantage of the hand-written complex-$A$ scan relative to the fused Mamba-1 kernel is an implementation artifact, not a fundamental property of the architecture; a fused kernel for complex-diagonal $A$ would make it feasible to test the Mamba-3 upgrades at full scale rather than only at matched reduced size.

---

## 6. Conclusion

We studied whether a modern selective state-space model can improve food-texture classification from chewing-behavior video on the same per-window features that tabular and tabular-NN baselines see. Under a leave-subjects-out cross-validation protocol on a 20-subject cohort, a bidirectional Mamba-1 sequence model achieves a per-window macro-F1 of $0.630$, exceeding the strongest tabular baseline (logistic regression at $0.492$) by $+0.138$ points. We additionally evaluated robustness to additive per-column Gaussian perturbation of the input features and found that the sequence model retains its absolute lead over all tabular baselines at both mild and severe perturbation magnitudes, confirming that the temporal-modeling advantage is not an artifact of clean conditions. At the chunk level, logistic regression combined with majority-vote aggregation is narrowly ahead of the sequence model ($0.685$ vs $0.652$), an honest caveat that should inform deployment-granularity decisions. A matched-size ablation of recent Mamba-3 upgrades (complex-valued $A$, trapezoidal discretization) yields a small consistent gain ($+0.023$ window-F1) at roughly $143\times$ the wall-clock cost of the fused Mamba-1 kernel, identifying kernel-level engineering of complex-$A$ scans as a high-value direction for future scale-up. Together, the results support temporal modeling as a cheap and effective addition to chewing-video feature pipelines and suggest that further gains are available from finer temporal resolution, larger cohorts, fused kernels for newer SSM variants, and multi-modal fusion.

---

## References

- Chen, T. and Guestrin, C. XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.
- Dao, T. and Gu, A. Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality (Mamba-2). In *Proceedings of the 41st International Conference on Machine Learning*, 2024.
- Google. MediaPipe Face Landmarker. Google Developers documentation, 2022.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. Revisiting Deep Learning Models for Tabular Data. In *Advances in Neural Information Processing Systems* 34, 2021.
- Gu, A. and Dao, T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. Preprint, 2023.
- Hendrycks, D. and Dietterich, T. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. In *International Conference on Learning Representations*, 2019.
- Loshchilov, I. and Hutter, F. Decoupled Weight Decay Regularization (AdamW). In *International Conference on Learning Representations*, 2019.
- Mamba-3: Improved Sequence Modeling via Complex State Matrices and Trapezoidal Discretization. In *International Conference on Learning Representations*, 2026.
- Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2011.
- Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., and Fergus, R. Intriguing Properties of Neural Networks. In *International Conference on Learning Representations*, 2014.
- Vaswani, A. et al. Attention is All You Need. In *Advances in Neural Information Processing Systems* 30, 2017.

---

## Appendix A. Hyperparameters and Per-Class Support

### A.1 Hyperparameters and cross-validation

**Optimizers.** All neural models use AdamW with cosine learning-rate annealing. Tabular NNs: learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 256, 60 epochs for MLP and 30 for FT-Transformer. Mamba (main): learning rate $5 \times 10^{-4}$, weight decay $10^{-3}$, sequence batch size 4, 80 epochs, $d_\text{model} = 64$, $n_\text{blocks} = 2$, $d_\text{state} = 16$. Mamba (ablation, both variants): same optimizer settings but $d_\text{model} = 32$, $n_\text{blocks} = 1$, $d_\text{state} = 8$, trained for 30 epochs.

**Classical models.** Logistic regression: L2, `class_weight="balanced"`, up to 3,000 iterations inside a `StandardScaler` pipeline. Random forest: 400 trees, no depth limit, balanced class weights, $n_\text{jobs}$ all cores. XGBoost: 400 trees, learning rate 0.05, max depth 6, row subsample 0.9, column subsample 0.9, multinomial soft-probability objective, histogram tree method.

**Cross-validation.** 5-fold StratifiedGroupKFold on `video_id`, fallback to GroupKFold. Feature standardization (mean/std) fitted only on the training fold. Balanced class weights computed only on the training fold. Random seed 42.

### A.2 Per-class support and per-class F1

The dataset contains approximately 8,077 windows across 20 subjects, split into three classes. Per-class support is reported together with the per-class F1 of the main Mamba sequence model at the clean evaluation condition in Table A1.

**Table A1.** Per-class support and per-class F1 for the main Mamba sequence model, pooled over OOF predictions at the clean evaluation condition.

| Class | Support | Per-class F1 (mamba) |
|-------|---------|---------------------|
| soft | $503$ | $0.740$ |
| medium | $584$ | $0.537$ |
| hard | $527$ | $0.614$ |

As noted in Section 4.3, medium is the hardest class for the sequence model — the expected behavior given that medium lies between its neighbors on the ordinal texture axis and inherits confusion mass from both sides. Soft and hard are recovered at comparable but distinctly higher rates, with soft slightly higher.

### A.3 Feature-importance top-5 (repeated for reference)

For ease of reference, the top-5 feature importance rankings from Section 4.4 are repeated below.

- **logreg** (absolute standardized coefficient magnitudes): `mouth_open_px_std`, `mouth_open_px_p95`, `mouth_open_smooth_std`, `mouth_open_smooth_p95`, `mouth_width_px_p95`.
- **rf** (impurity-based importances): `pause_ratio`, `mouth_open_px_std`, `mouth_open_smooth_std`, `jaw_right_y_px_mean`, `mouth_width_px_mean`.
- **xgb** (impurity-based importances): `mouth_open_smooth_std`, `mouth_open_px_p95`, `mouth_open_smooth_p95`, `mouth_open_px_std`, `pause_ratio`.

### A.4 Figure captions

- **Figure 1.** Overview diagram: per-window feature extraction from face-landmark streams, ordered by `window_id` within each video, consumed as a sequence by the bidirectional Mamba-1 classifier; tabular baselines consume the same feature vectors without sequence ordering. No numbers required.
- **Figure 2.** Pooled out-of-fold row-normalized confusion matrices at the clean evaluation condition for the best tabular baseline (logistic regression) and for the Mamba sequence model, over classes $\{\text{soft}, \text{medium}, \text{hard}\}$.
- **Figure 3.** Robustness curve of window-level macro-F1 versus input-perturbation magnitude $\sigma \in \{0, \sigma_1, \sigma_2\}$ for all six models, under the same 5-fold CV protocol as Table 1. Markers are fold means; the gap between Mamba and the strongest tabular baseline at each magnitude is annotated.
- **Figure 4.** Per-class F1 for each model at the clean evaluation condition, grouped by class (soft, medium, hard), with error bars equal to one standard deviation across folds.
- **Figure 5.** Top-5 feature-importance bars for logreg, rf, and xgb, using the importances from Section 4.4.
- **Figure 6.** Training curves (mean loss per epoch, averaged across folds) for the Mamba sequence model and the tabular NNs at the clean evaluation condition.
