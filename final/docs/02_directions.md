# Step 2 — Three candidate modification directions

Prerequisite: [`01_baselines.md`](01_baselines.md). All targets are ICBHI 4-class, official 60/40
subject-independent split, Score = (Sp + Se)/2.

**Backbone decision that applies to all three directions: use BEATs.**
BEATs + plain cross-entropy scores **63.49** (Sp 78.77 / Se 48.21), versus AST + Patch-Mix at
62.37 and AST + CE at 59.55. Whichever direction we pick, starting from BEATs puts us above the
widely-cited SOTA before we do anything — so the honest claim we must defend is
*"our modification improves on BEATs + CE"*, i.e. **beat 63.49, not 62.37**.

---

## Direction A — Metadata-conditioned backbone

### The idea

BTS showed that ICBHI metadata carries real signal: audio-only CLAP 62.56 → BTS 63.54 (**+0.98**)
purely from encoding metadata as free text through CLAP's text tower. Their ablation:

| Removed | Score | Δ |
|---|---|---|
| — (all metadata) | 63.54 | — |
| no chest location | 62.76 | −0.78 |
| no device | 62.66 | −0.88 |
| no age/gender | 62.88 | −0.66 |
| audio only | 62.56 | −0.98 |

But this requires CLAP *because* the mechanism is a text encoder. **Nobody has injected metadata
directly into a non-CLAP backbone.** That is the gap: get BTS's gain on BEATs, without CLAP.

### What metadata ICBHI actually gives us

| Source file | Field | Cardinality | Safe to use? |
|---|---|---|---|
| filename | Chest location — Tc, Al, Ar, Pl, Pr, Ll, Lr | 7 | ✅ |
| filename | Recording device — AKGC417L, LittC2SE, Litt3200, Meditron | 4 | ✅ |
| filename | Acquisition mode — sequential (sc) / multichannel (mc) | 2 | ✅ |
| `demographic_info.txt` | Age, Sex | cont. + 2 | ✅ |
| `demographic_info.txt` | Adult BMI, child weight/height | cont. | ✅ (sparse) |
| `patient_diagnosis.csv` | Diagnosis (COPD, asthma, …) | 8 | ❌ **never** |

> ⚠ **Do not use `patient_diagnosis.csv` as a feature.** Diagnosis almost determines whether
> crackles/wheezes are present. Including it inflates the score and invalidates the whole
> comparison. This is the most likely way to accidentally produce a fake 90 % result.

### Implementation on BEATs

Three options, increasing invasiveness:

1. **Late concat** — embed each field, concat to the pooled BEATs embedding before the classifier.
   Simplest; closest to BTS's `concat(z_t, z_a)`. ~30 lines.
2. **FiLM conditioning** — metadata vector → (γ, β), applied as `γ ⊙ h + β` on the last *k*
   transformer blocks. Modulates the audio representation rather than sitting beside it.
   Stronger, still cheap, doesn't disturb pretrained weights at init if γ→1, β→0.
3. **Metadata tokens** — prepend learned metadata embeddings as extra tokens in the BEATs
   sequence, so self-attention sees them at every layer. Most expressive, most likely to
   destabilise pretrained attention.

Recommend implementing **1 as a sanity check, then 2 as the real method**, and reporting both —
that ablation is itself a contribution.

### Honest risks

- **Device is a confound, not just a feature.** The 4 stethoscopes were used on different patient
  cohorts, so device partially encodes the label distribution. The model may learn the shortcut
  rather than acoustics. It will still raise the test score (test set shares the confound), but
  it is scientifically weak — and SG-SCL argues the opposite, that device variation should be
  *removed*. Expect a reviewer to ask about this. Mitigation: report per-device breakdown.
- BTS's +0.98 may not transfer; BEATs may already encode some of it implicitly.

### Verdict

| | |
|---|---|
| Expected gain | +0.5 to +1.5 → **~64.0–65.0** |
| Implementation cost | **Low** (~half a day) |
| Compute cost | 1 config × 3 seeds ≈ 4 h |
| Novelty | **Medium-high** — "metadata-conditioned BEATs via FiLM" is genuinely unpublished |
| Risk | Low-medium (the device confound is a discussion point, not a blocker) |

---

## Direction B — Sensitivity-targeted / Score-aligned objective

### The idea

Look down the Score column of the baseline table: **every method is specificity-heavy.**

| Method | Sp | Se | gap |
|---|---|---|---|
| SE+SA | 81.25 | 17.84 | 63.4 |
| AST + Patch-Mix | 81.66 | 43.07 | 38.6 |
| BTS | 81.40 | 45.67 | 35.7 |
| BEATs + CE | 78.77 | 48.21 | 30.6 |
| Lung-SRAD | 79.53 | 49.42 | 30.1 |

Nobody has reported Se above 50 on this benchmark. Yet the metric weights Sp and Se **equally**,
while the data is 3,642 normal vs 3,256 abnormal cycles *pooled* — and within abnormal, 1,864 /
886 / 506. Everyone trains with plain 4-way cross-entropy, which optimises neither the metric nor
the class balance.

### The key observation

The ICBHI Score is exactly a **two-group balanced accuracy**:

```
Score = ½·recall(normal) + ½·recall(abnormal-pooled)
```

So the loss that matches the metric is not 4-class-balanced CE and not focal loss — it is
**group-balanced CE**, where sample weights are

```
w_i = ½ / N_normal          if y_i = normal
w_i = ½ / N_abnormal_pooled if y_i ∈ {crackle, wheeze, both}
```

with an optional second level inside the abnormal group to stop "both" (506 cycles) vanishing.
This is a direct, principled surrogate for the evaluation metric, and I have not found it stated
this way in any ICBHI paper — everyone reaches for generic long-tail losses instead.

Layered on top, essentially for free:

- **Logit adjustment** (Menon et al.): subtract `τ·log π_y` from logits at train time, or add it
  post-hoc at test time. Post-hoc costs *zero* extra training and directly slides the Sp/Se
  operating point.
- **Validation-time prior search**: sweep a per-class logit bias on the validation split to
  maximise Score. Pure post-processing, minutes of CPU. Almost certainly worth +0.3–1.0 on its
  own, and it is the sort of thing that is embarrassing to leave on the table.
- **LDAM / angular margin**: QLung's log-scaled class-imbalance margin gives **+2.46 over AST+CE**.

### Existing evidence

| Evidence | Result |
|---|---|
| QLung log-scaled margin on AST | 59.55 → 62.01 (**+2.46**) |
| CNN-LSTM + focal loss, Sensors 2022 | Se 47.37 / Sp 82.46 / **64.92** — see caveat |
| LDAM + focal + effective-number ensemble | macro-F1 0.46 (different protocol) |

⚠ The Sensors 2022 number claims the official 60/40 split and would have beaten all published
work until 2025, yet no later paper cites it. Before building on it, spend 30 minutes checking
their cycle extraction. If it holds up, it is strong evidence for this direction; if it doesn't,
it is a good cautionary example for the report.

### Honest risks

- **This is the direction most likely to net out to zero.** Trading Sp for Se moves along a
  curve; the Score only improves if the *curve itself* moves, not just the operating point.
  Post-hoc threshold tuning finds the best point on the existing curve — a real but bounded gain.
- Reviewers may call focal/CB loss "standard practice, not a contribution". The
  Score-aligned group-balanced formulation is the part that is defensible; frame it that way.
- Aggressive reweighting on a 6.9k-sample dataset can destabilise fine-tuning.

### Verdict

| | |
|---|---|
| Expected gain | +0.5 to +2.0 → **~64.0–65.5** |
| Implementation cost | **Very low** (~2 hours; no architecture change) |
| Compute cost | Cheapest — post-hoc variants need no retraining at all |
| Novelty | **Medium** as framed (Score-aligned objective); Low if presented as "we used focal loss" |
| Risk | Medium — the most likely to produce a null result |

---

## Direction C — Frequency-aware architecture

### The idea

The two pathologies are physically different in the time-frequency plane, and a uniform 16×16
patch grid treats both axes identically:

| | Crackles | Wheezes |
|---|---|---|
| Duration | 5–15 ms, discontinuous | > 100 ms, continuous |
| Spectrum | broadband transient | narrow-band tonal + harmonics |
| Band | ~100–2000 Hz | ~100–1000 Hz fundamental |
| Signature | vertical striping in the spectrogram | horizontal striping |

So crackle evidence lives along the **time** axis and wheeze evidence along the **frequency**
axis. That asymmetry is a genuine, physiologically-grounded inductive bias — and it is exactly
what the current frontier exploits.

### What already works

| Method | Mechanism | Result |
|---|---|---|
| Lung-SRAD | dual-axis Patch-Mix (separate temporal and frequency band mixing) + Gaussian spectral regularisation | 61.06 → **64.48** (+3.4) |
| ADD-RSC (Interspeech 2025) | learnable spectral masks + soft shrinkage + differential attention | **65.53** |
| Freq. selection + attention | frequency band selection module | reported +2.16 over baseline |

Note Lung-SRAD reaches the **highest sensitivity in the entire table (49.42)** — consistent with
the claim that frequency-axis structure is where the abnormal-class signal is.

### Implementation on BEATs

- **Cheap version — dual-axis Patch-Mix as pure augmentation.** Mix contiguous *frequency bands*
  between samples, and separately contiguous *time segments*, instead of random patches. No
  architecture change at all, just a data transform + the mixed-label contrastive loss. This is
  ~80 % of Lung-SRAD's reported gain for ~20 % of the work.
- **Medium version — learnable spectral gate.** A per-mel-bin learnable mask (or a small
  band-attention module) applied to the log-mel input before BEATs' patch embedding. Initialise
  to identity so pretrained weights are undisturbed.
- **Expensive version — axis-factorised attention** inside the encoder. Breaks BEATs pretraining;
  not advisable for this project.

### Honest risks

- **Highest risk of the three.** Anything that touches the patch embedding fights ~90 M
  pretrained parameters, and ICBHI's 4,142 training cycles are far too few to re-learn them.
- Frequency-band mixing can destroy the very narrow-band structure that defines a wheeze — the
  augmentation could hurt. Needs a band-width hyperparameter sweep, i.e. more compute.
- Lung-SRAD's gain is reported on a state-space backbone whose spectral behaviour differs from
  BEATs (the paper explicitly notes AST is low-pass while DASS is not). The gain may not transfer.

### Verdict

| | |
|---|---|
| Expected gain | −0.5 to +2.5 → **~63.0–66.0** (widest spread) |
| Implementation cost | **Medium-high** (2–3 days for the medium version) |
| Compute cost | Highest — needs a hyperparameter sweep |
| Novelty | **High** |
| Risk | **High** |

---

## Side-by-side

| | A · Metadata | B · Score-aligned loss | C · Frequency-aware |
|---|---|---|---|
| Starting point | BEATs + CE = 63.49 | BEATs + CE = 63.49 | BEATs + CE = 63.49 |
| Expected result | 64.0 – 65.0 | 64.0 – 65.5 | 63.0 – 66.0 |
| Confidence in a positive result | **High** | Medium | Medium-low |
| Dev time | ~0.5 day | ~2 h | 2–3 days |
| GPU hours | ~4 | ~2 | ~12+ |
| Novelty | Medium-high | Medium | High |
| Main risk | device confound | nets out to zero | breaks pretrained weights |
| Prior evidence of the gain | +0.98 (BTS, on CLAP) | +2.46 (QLung, on AST) | +3.4 (Lung-SRAD, on DASS) |

## Recommendation

**Do A + B together, keep C as a stretch goal.**

They are orthogonal — A adds information the audio doesn't carry, B fixes the objective's
mismatch with the metric — and neither touches the pretrained encoder, so they compose without
fighting each other. Combined they are still under a day of implementation and ~6 GPU-hours,
which fits Colab/Kaggle session limits comfortably.

Proposed final model: **MC-BEATs** — metadata-conditioned BEATs (FiLM) trained with a
Score-aligned group-balanced objective and post-hoc prior calibration.

Ablation ladder for the notebook, each row one training run, 3 seeds:

| Row | Config | Expected |
|---|---|---|
| 0 | BEATs + CE *(reproduce, anchors the cited table)* | ~63.5 |
| 1 | + Score-aligned group-balanced loss | ~64.0 |
| 2 | + post-hoc prior calibration *(no retrain)* | ~64.3 |
| 3 | + metadata late-concat | ~64.5 |
| 4 | + metadata FiLM = **MC-BEATs** | **~64.5–65.0** |
| 5 | + dual-axis Patch-Mix *(direction C, stretch)* | ~65+ |

This gives a clean ablation table where every row is one controlled change — which is what makes
a benchmark notebook convincing, independent of the final number.

**Fallback if the gains don't materialise:** rows 0–2 alone still beat the 62.37 reference, and a
well-executed negative result on metadata (with the device-confound analysis) is a legitimate
finding for the writeup.
