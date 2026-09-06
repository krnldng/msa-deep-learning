# Step 1 — ICBHI 2017 Baseline Survey (2021–2026)

Goal: pick reproducible public baselines for the ICBHI 2017 respiratory sound challenge, then
beat them with an architecture/method modification. This document is the reference the
benchmark notebook will be built from.

---

## 1. The task and the evaluation protocol (read this first)

**Dataset.** ICBHI 2017 Respiratory Sound Database — 920 annotated recordings, 126 subjects,
~5.5 h audio, split into **6,898 breathing cycles**:

| Class | Cycles |
|---|---|
| Normal | 3,642 |
| Crackle | 1,864 |
| Wheeze | 886 |
| Both (crackle+wheeze) | 506 |

Two tasks are reported in the literature:
- **4-class**: normal / crackle / wheeze / both ← *this is the standard benchmark, use this*
- **2-class**: normal / abnormal

**Metrics** (these are non-standard, do not use plain accuracy):

```
Specificity Sp = correctly classified Normal cycles / all Normal cycles
Sensitivity Se = correctly classified abnormal cycles / all abnormal cycles   (crackle+wheeze+both pooled)
ICBHI Score    = (Sp + Se) / 2
```

**⚠ The single biggest comparability trap.** Published ICBHI numbers range from ~50 % to ~99 %
and most of that spread is *protocol*, not method quality. Three different protocols circulate:

| Protocol | Typical Score | Comparable? |
|---|---|---|
| **Official 60/40 split** (`official_split.txt`, near-subject-disjoint) | 50–65 % | ✅ **Use this** |
| Random 80/20 split, subject-disjoint | 65–75 % | ⚠ different paper family |
| Random split at *cycle* level (same patient in train+test) | 90–99 % | ❌ patient leakage, meaningless |

Any paper claiming ">95 % accuracy on ICBHI" is almost certainly leaking patients across the
split.

> ⚠ **Verified caveat on the official split itself.** It is *near*-subject-disjoint, not fully:
> patients **156** and **218** have recordings on both sides (124/126 are clean). Confirmed by
> parsing `official_split.txt` directly. Every cited paper uses the file as-is, so this inflates
> all published numbers equally and does not distort method-to-method comparison — but it must be
> stated when reporting absolute values, and the split must **not** be "fixed" or comparability
> is lost. All numbers in the table below are the **official 60/40 subject-independent split**
unless flagged otherwise. Our notebook must use the same split or the comparison is void.

---

## 2. Baselines worth using

Ordered by how useful they are as a starting point for you.

### Tier A — the reference line to beat

**A1. Bae et al., Patch-Mix Contrastive Learning + AST — Interspeech 2023**
- Paper: https://arxiv.org/abs/2305.14032 · [ISCA archive](https://www.isca-archive.org/interspeech_2023/bae23b_interspeech.html)
- Code: https://github.com/raymin0223/patch-mix_contrastive_learning
- Result: **Sp 81.66 / Se 43.07 / Score 62.37** (4-class), 68.71 (2-class)
- Backbone: Audio Spectrogram Transformer (86 M), pretrained ImageNet + AudioSet
- Why it matters: this is *the* reference implementation the whole 2023–2026 line of work forks
  from. Its repo also contains clean, correct implementations of the official split, the metric,
  and the CE fine-tuning baseline. **Start here.**

**A2. Bae et al., plain AST fine-tuning** (ablation row in the same repo/paper)
- Result: **Sp 77.14 / Se 41.97 / Score 59.55**
- This is the honest "no tricks" transformer baseline — the cleanest thing to put in the notebook
  as *the* baseline, with Patch-Mix as the strong baseline.

### Tier B — light, laptop-friendly baselines

**B1. Moummad & Farrugia, Supervised Contrastive Learning — WASPAA 2023**
- Paper: https://arxiv.org/abs/2210.16192 (see also the metadata/M-SCL version)
- Code: https://github.com/ilyassmoummad/scl_icbhi2017
- Result (paper): CE **Score 55.24** · SCL **Score 57.55** · README M-SCL **Sp 76.93 / Se 39.15 / Score 58.04** (10-run mean)
- Backbone: **CNN6 / CNN10 / CNN14 (PANNs)** — 5–80 M params, trains fast
- Why it matters: **this is the one that will comfortably fit your 6 GB RTX 4050.** Good for a
  from-scratch-ish baseline and for fast iteration on ideas before you spend GPU time on AST.

**B2. RespireNet — EMBC 2021** (pre-2022 but still the most-cited CNN baseline)
- Paper: https://arxiv.org/abs/2011.00196 · [PDF](https://mohitjaindr.github.io/pdfs/c30-embc-2021.pdf)
- Code: https://github.com/microsoft/RespireNet
- Result: **Sp 72.30 / Se 40.10 / Score 56.20** (CBA+BRC+FT, 4-class, 60/40)
- Backbone: ResNet34 (ImageNet). Contributes concatenation-based augmentation, device-specific
  fine-tuning. Included because reviewers expect to see it.

### Tier C — stronger 2024–2025 methods (the target band)

**C1. Kim et al., Stethoscope-guided Supervised Contrastive Learning (SG-SCL) — ICASSP 2024**
- Paper: https://arxiv.org/abs/2312.09603
- Code: https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning
- Result: **Sp 79.87 / Se 43.55 / Score 61.71**
- Idea: treat the 4 recording devices as domains, contrastive alignment across stethoscopes.

**C2. Kim et al., RepAugment — Interspeech 2024**
- Paper: https://arxiv.org/abs/2405.02996
- Code: https://github.com/RSC-Toolkit/RepAugment
- Result: **Sp 82.47 / Se 40.55 / Score 61.51**
- Idea: augmentation in *representation* space, not input space (input-agnostic).

**C3. Kim et al., BTS: Bridging Text and Sound — Interspeech 2024**
- Paper: https://arxiv.org/abs/2406.06786
- Code: https://github.com/kaen2891/bts
- Result: **Sp 81.40 ± 2.57 / Se 45.67 ± 2.66 / Score 63.54 ± 0.80**
- Idea: CLAP text-audio model; encode metadata (age, sex, device, chest location) as free text.
- Note: CLAP is heavier than AST — check VRAM before committing.

**C4. Kim et al., AFT — adversarial fine-tuning with generated sounds — NeurIPS 2023 W-DGM4H**
- Code: https://github.com/kaen2891/adversarial_fine-tuning_using_generated_respiratory_sound
- Result: **Sp 80.72 / Se 42.86 / Score 61.79** (Mixed-500)

**C5. Toikkanen et al., Ensemble Knowledge Distillation — Interspeech 2025**
- Paper: https://arxiv.org/abs/2505.22027 · [ISCA PDF](https://www.isca-archive.org/interspeech_2025/toikkanen25_interspeech.pdf)
- Code: https://github.com/RSC-Toolkit/rsc-ensemble-kd
- Result: **Sp 82.89 / Se 45.90 / Score 64.39** (BTS-d, k=15); 65.45 with a 2nd-gen ensemble
- Idea: architecture-agnostic soft-label distillation from an ensemble of teachers. Cheap to
  apply on top of *any* baseline — worth stealing for our modification.

### Tier C+ — ⭐ the important find: BEATs

**C6. Yang et al., PAFA (Patient-Aware Feature Alignment) — 2025**
- Paper: https://arxiv.org/abs/2505.23834
- Code: https://github.com/wa976/PAFA
- Results: **BEATs + plain CE → Sp 78.77 / Se 48.21 / Score 63.49**
  · **BEATs + PAFA → Sp 82.05 / Se 47.63 / Score 64.84**
- Idea: two losses — patient cohesion-separation + global patient alignment — to remove
  patient-identity variance from the feature space.

> **This is the single most actionable finding in the survey.** A **BEATs** backbone with
> *nothing but cross-entropy* scores **63.49** — higher than AST + Patch-Mix (62.37) and
> level with BTS (63.54), at AST-like cost and with no contrastive machinery. Almost the
> entire 2023–2024 AST literature is bottlenecked by its backbone, not its method.
> **Recommendation: build on BEATs, not AST.** It raises our floor by ~4 points for free.

### Tier D — 2025–2026 frontier (how much headroom is left)

| Method | Sp | Se | Score | Paper | Code |
|---|---|---|---|---|---|
| QLung (quality-adaptive angular margin, CLAP) | 81.98 | 44.81 | 63.39 | [arXiv:2606.11915](https://arxiv.org/abs/2606.11915) | [QLung](https://github.com/RSC-Toolkit/QLung) |
| Lung-SRAD (audio state-space model + dual-axis Patch-Mix) | 79.53 | 49.42 | 64.48 | [arXiv:2606.11922](https://arxiv.org/abs/2606.11922) | [Lung-SRAD](https://github.com/RSC-Toolkit/Lung-SRAD) |
| **PAFA (BEATs)** | 82.05 | 47.63 | **64.84** | [arXiv:2505.23834](https://arxiv.org/abs/2505.23834) | [PAFA](https://github.com/wa976/PAFA) |
| ADD-RSC (adaptive differential denoising, Interspeech 2025) | — | — | 65.53 | [arXiv:2506.02505](https://arxiv.org/abs/2506.02505) | [ADD-RSC](https://github.com/deegy666/ADD-RSC) |
| BTS++ (k=30 ensemble KD) | — | — | 65.69 | [arXiv:2505.22027](https://arxiv.org/abs/2505.22027) | [rsc-ensemble-kd](https://github.com/RSC-Toolkit/rsc-ensemble-kd) |
| Meta-Ensemble w/ diverse data splits (2026) | — | — | **66.49** | [arXiv:2604.24096](https://arxiv.org/html/2604.24096v1) | — |
| BTS-CARD (counterfactual debiasing, ICASSP 2026) | — | — | — | [arXiv:2510.22263](https://arxiv.org/html/2510.22263) | [BTS-CARD](https://github.com/RSC-Toolkit/BTS-CARD) |

⚠ **Two numbers to treat with suspicion, not to copy into a comparison table:**
- *SAM-optimized AST*, Score 68.10 ([arXiv:2512.22564](https://www.arxiv.org/pdf/2512.22564)) — an
  outlier far above the whole 62–66 cluster and absent from every peer comparison table.
  Protocol unverified.
- *CNN-LSTM + focal loss*, Se 47.37 / Sp 82.46 / **Score 64.92** (Sensors 2022,
  [PMC8838187](https://pmc.ncbi.nlm.nih.gov/articles/PMC8838187/)). The paper explicitly claims
  the official 60/40 cycle-level split. If true, a small CNN-LSTM with focal loss beat everything
  published until 2025 — and no subsequent paper cites it. Either the cycle extraction differs,
  or it is a genuinely under-recognised result. **Directly relevant to Direction B; worth a
  30-minute check rather than a citation.**

**Realistic ceiling on the official split as of Sept 2026: ~65–66.5**, and the top of that range
costs 30× compute (ensembles). Single-model SOTA is ~64.5–65.5. Anything we build landing in
**63–65 is a legitimately competitive single-model result.**

---

## 3. Comparison table (all: ICBHI 4-class, official 60/40 subject-independent split)

| # | Method | Year | Venue | Backbone | Pretrain | Sp % | Se % | **Score %** | Code |
|---|---|---|---|---|---|---|---|---|---|
| 1 | SE+SA | 2020 | Interspeech | ResNet18 | — | 81.25 | 17.84 | 49.55 | — |
| 2 | Ren et al. (prototype) | 2022 | ICASSP | CNN8-Pt | — | 72.96 | 27.78 | 50.37 | — |
| 3 | LungRN+NL | 2020 | Interspeech | ResNet-NL | — | 63.20 | 41.32 | 52.26 | — |
| 4 | Chang et al. | 2022 | — | CNN8-dilated | — | 69.92 | 35.85 | 52.89 | — |
| 5 | Moummad et al. (CE) | 2023 | WASPAA | CNN6 | AudioSet | 70.09 | 40.39 | 55.24 | [link](https://github.com/ilyassmoummad/scl_icbhi2017) |
| 6 | Wang et al. (Splice) | 2022 | — | ResNeSt | ImageNet | 70.40 | 40.20 | 55.30 | — |
| 7 | **RespireNet** (CBA+BRC+FT) | 2021 | EMBC | ResNet34 | ImageNet | 72.30 | 40.10 | 56.20 | [link](https://github.com/microsoft/RespireNet) |
| 8 | Late-Fusion | 2022 | — | Inc-03 + VGG14 | ImageNet | 85.60 | 30.00 | 57.30 | — |
| 9 | Nguyen & Pernkopf (StochNorm) | 2022 | IEEE TBME | ResNet50 | ImageNet | 78.86 | 36.40 | 57.63 | [paper](https://arxiv.org/abs/2108.01991) |
| 10 | **Moummad et al. (SCL)** | 2023 | WASPAA | CNN6 | AudioSet | 75.95 | 39.15 | 57.55 | [link](https://github.com/ilyassmoummad/scl_icbhi2017) |
| 11 | Nguyen & Pernkopf (Co-Tuning) | 2022 | IEEE TBME | ResNet50 | ImageNet | 79.34 | 37.24 | 58.29 | [paper](https://arxiv.org/abs/2108.01991) |
| 12 | **Bae et al. (AST fine-tune)** | 2023 | Interspeech | AST | IN+AS | 77.14 | 41.97 | **59.55** | [link](https://github.com/raymin0223/patch-mix_contrastive_learning) |
| 13 | Kim et al. (RepAugment) | 2024 | Interspeech | AST | IN+AS | 82.47 | 40.55 | 61.51 | [link](https://github.com/RSC-Toolkit/RepAugment) |
| 14 | Kim et al. (SG-SCL) | 2024 | ICASSP | AST | IN+AS | 79.87 | 43.55 | 61.71 | [link](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning) |
| 15 | Kim et al. (AFT, Mixed-500) | 2023 | NeurIPS-W | AST | IN+AS | 80.72 | 42.86 | 61.79 | [link](https://github.com/kaen2891/adversarial_fine-tuning_using_generated_respiratory_sound) |
| 16 | **Bae et al. (Patch-Mix CL)** | 2023 | Interspeech | AST | IN+AS | 81.66 | 43.07 | **62.37** | [link](https://github.com/raymin0223/patch-mix_contrastive_learning) |
| 17 | Audio-CLAP fine-tune | 2024 | Interspeech | CLAP | LAION-630K | 80.85 | 44.67 | 62.56 | [link](https://github.com/kaen2891/bts) |
| 18 | **BTS** | 2024 | Interspeech | CLAP | LAION-630K | 81.40 | 45.67 | **63.54** | [link](https://github.com/kaen2891/bts) |
| 19 | BTS-d (ensemble KD, k=15) | 2025 | Interspeech | CLAP | LAION-630K | 82.89 | 45.90 | **64.39** | [link](https://github.com/RSC-Toolkit/rsc-ensemble-kd) |
| 20 | BTS-d++ (2nd-gen ensemble) | 2025 | Interspeech | CLAP | LAION-630K | 88.09 | 42.82 | 65.45 | [link](https://github.com/RSC-Toolkit/rsc-ensemble-kd) |
| 21 | QLung | 2026 | arXiv | CLAP | LAION-630K | 81.98 | 44.81 | 63.39 | [link](https://github.com/RSC-Toolkit/QLung) |
| 22 | **BEATs + CE** ⭐ | 2025 | arXiv | BEATs | AudioSet | 78.77 | 48.21 | **63.49** | [link](https://github.com/wa976/PAFA) |
| 23 | Lung-SRAD | 2026 | arXiv | DASS (SSM) | AudioSet | 79.53 | 49.42 | 64.48 | [link](https://github.com/RSC-Toolkit/Lung-SRAD) |
| 24 | **PAFA** | 2025 | arXiv | BEATs | AudioSet | 82.05 | 47.63 | **64.84** | [link](https://github.com/wa976/PAFA) |
| 25 | ADD-RSC | 2025 | Interspeech | — | — | — | — | 65.53 | [link](https://github.com/deegy666/ADD-RSC) |
| 26 | BTS++ (k=30 ensemble) | 2025 | Interspeech | CLAP | LAION-630K | — | — | 65.69 | [link](https://github.com/RSC-Toolkit/rsc-ensemble-kd) |
| 27 | Meta-Ensemble (diverse splits) | 2026 | arXiv | — | — | — | — | **66.49** | — |

Rows 1–20 sourced from the comparison tables in the BTS paper (arXiv:2406.06786) and the
Ensemble-KD paper (arXiv:2505.22027), which share a consistent protocol; rows 21–27 from the
QLung, PAFA and Lung-SRAD papers, which reproduce the same protocol and the same rows 1–20.

### What the table actually tells us

1. **Specificity is easy, sensitivity is the bottleneck.** Every method sits at Sp 70–88 but
   Se 18–46. The score is dragged down entirely by missed abnormal cycles. A modification that
   trades a little Sp for Se is the highest-leverage move.
2. **Row 1 is the cautionary tale**: Sp 81.25 / Se 17.84 — a model that essentially predicts
   "normal". Watch for this failure mode in our own runs.
3. **Pretraining dominates architecture.** Every method above 59 uses a large-scale audio-
   pretrained backbone (AudioSet/LAION). Training from scratch on 6.9k cycles caps out ~52.
4. **The gains since 2023 come from the training recipe, not new architectures** — mixing,
   contrastive objectives, metadata, distillation. That is where our modification should live.

---

## 4. Compute plan — Colab Pro (paid) + Kaggle (decided)

All runs execute on **Colab Pro** (paid) or Kaggle, not the local RTX 4050 (6 GB). VRAM is no
longer a ceiling at all — Pro offers L4 (24 GB) and A100 (40 GB), so even the CLAP/BTS tier is
reachable.

| Backbone | Params | 6 GB local | Colab Pro / Kaggle | Est. time / 50 epochs |
|---|---|---|---|---|
| CNN6 (PANNs) | ~5 M | ✅ | ✅ | ~10 min |
| ResNet34 | 21 M | ✅ | ✅ | ~15 min |
| AST-base | 86 M | ⚠ bs ≤ 8 + AMP | ✅ bs 16–32 | ~45–90 min |
| **BEATs** | ~90 M | ⚠ | ✅ | ~45–90 min |
| CLAP (BTS) | 150 M+ | ❌ | ✅ | ~1.5–2 h |

**The binding constraint is now compute units, not VRAM or session length.** Colab Pro grants
100 CU/month; burn rate is roughly T4 ~1.2–1.8 CU/h, A100 ~5–15 CU/h (public sources disagree
and Google does not publish official rates — read the in-session CU meter after the first run).

- **Budget.** The §6 ablation is 6 rows × 3 seeds = 18 runs ≈ 27 GPU-hours ≈ 32–49 CU on T4,
  i.e. one third to one half of the monthly allowance. Comfortable.
- **⚠ Do not pick A100 for this workload.** A 90 M model on 4,142 samples will not saturate it;
  you pay 4–8× the CU for far less than 4–8× the speed. **Use T4 or L4.** Reserve A100 for the
  CLAP tier if it is ever attempted.
- **Running out of CU mid-run drops you to the free tier immediately** — 12 h cap, 90 min idle
  timeout — so checkpoint to Drive every epoch regardless of tier.
- **Background execution is Pro+, not Pro.** With plain Pro the browser tab must stay open.
  Kaggle's "Save & Run All" does run detached (30 GPU-h/week), so: **Colab Pro for interactive
  development, Kaggle for long overnight runs.**
- **Dataset re-download.** ICBHI is ~2 GB. Persist it (Kaggle Dataset, or Drive mount) rather
  than re-downloading per session. The notebook must cache preprocessed cycles.
- **Determinism.** Report mean ± std over ≥ 3 seeds. Published std on this benchmark is
  ±0.3–1.3, so a single-seed "win" of < 1 point is not a win.

## 5. Benchmark design — cited baselines (decided)

Per your decision, the notebook **cites** published baselines rather than retraining them, and
trains only our model. That is standard practice in this literature, but it is only valid if our
harness matches theirs exactly. The notebook will therefore **assert** the protocol rather than
assume it:

| Assertion | Expected value |
|---|---|
| Split file | official `ICBHI_challenge_train_test.txt` |
| Split granularity | subject-level, no patient in both sides |
| Train cycles | **4,142** |
| Test cycles | **2,756** |
| Class counts (all) | normal 3,642 · crackle 1,864 · wheeze 886 · both 506 |
| Metric | Score = (Sp + Se)/2, Se pooled over the 3 abnormal classes |
| Cycle extraction | per-cycle from `.txt` annotations, 8 s window (AST/BEATs convention) |

If any assertion fails, the cited comparison is void and the notebook says so loudly. This is
the safeguard that makes "cite, don't retrain" defensible.

**Cited reference line (the numbers we must beat):**

| Role | Method | Score |
|---|---|---|
| Classic CNN baseline | RespireNet | 56.20 |
| Transformer baseline | AST + CE | 59.55 |
| Widely-cited SOTA | AST + Patch-Mix CL | **62.37** |
| Strong modern baseline | BEATs + CE | **63.49** |
| Best comparable single model | PAFA (BEATs) | 64.84 |
| Best at any cost (30× compute) | BTS++ ensemble | 65.69 |

**Success criterion:** our single model should clear **62.37** to claim "surpasses the widely-cited
SOTA", and ideally clear **63.49** so the win is not merely a backbone swap.

> ⚠ Honest framing note: if we train on BEATs, beating 62.37 proves nothing by itself — BEATs+CE
> already does that. The claim must be *"our modification improves over BEATs+CE (63.49) on the
> same backbone"*. The notebook keeps a switch to train BEATs+CE ourselves so this delta is
> measured, not cited — cheap (one extra run) and it is what makes the result credible.

## 6. Where the headroom is — the three candidate directions

Detailed analysis in [`02_directions.md`](02_directions.md). Summary of the slack visible in the
table above:

1. **Sensitivity is the bottleneck.** Sp 70–88 vs Se 18–49 across every single method. Note that
   the newer high scorers (Lung-SRAD Se 49.4, PAFA Se 47.6, BEATs+CE Se 48.2) win precisely by
   lifting Se, not Sp.
2. **Backbone > method.** BEATs+CE (63.49) beats four years of AST-based method engineering.
3. **Recipe > architecture.** Post-2023 gains come from mixing, contrastive objectives, metadata
   and distillation — cheap to implement, no new architecture required.
4. **Metadata is unexploited outside CLAP.** BTS gets +0.98 from age/sex/device/location via a
   text encoder; nobody has injected it directly into a non-CLAP backbone.
5. **Patient identity is a confound.** PAFA's +1.35 comes purely from removing patient-specific
   variance — a signal orthogonal to everything else in the table.

## Sources

- [Patch-Mix Contrastive Learning with AST (Interspeech 2023)](https://arxiv.org/abs/2305.14032) · [code](https://github.com/raymin0223/patch-mix_contrastive_learning) · [ISCA](https://www.isca-archive.org/interspeech_2023/bae23b_interspeech.html)
- [BTS: Bridging Text and Sound Modalities (Interspeech 2024)](https://arxiv.org/abs/2406.06786) · [code](https://github.com/kaen2891/bts)
- [Improving RSC with Architecture-Agnostic Knowledge Distillation from Ensembles (Interspeech 2025)](https://arxiv.org/abs/2505.22027) · [code](https://github.com/RSC-Toolkit/rsc-ensemble-kd) · [ISCA PDF](https://www.isca-archive.org/interspeech_2025/toikkanen25_interspeech.pdf)
- [Stethoscope-guided Supervised Contrastive Learning (ICASSP 2024)](https://arxiv.org/abs/2312.09603) · [code](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning)
- [RepAugment (Interspeech 2024)](https://arxiv.org/abs/2405.02996) · [code](https://github.com/RSC-Toolkit/RepAugment)
- [RespireNet (EMBC 2021)](https://arxiv.org/abs/2011.00196) · [code](https://github.com/microsoft/RespireNet)
- [Pretraining Respiratory Sound Representations using Metadata and Contrastive Learning (WASPAA 2023)](https://github.com/ilyassmoummad/scl_icbhi2017)
- [Lung Sound Classification Using Co-tuning and Stochastic Normalization (IEEE TBME 2022)](https://arxiv.org/abs/2108.01991)
- [Lung-SRAD (arXiv 2606.11922)](https://arxiv.org/abs/2606.11922) · [code](https://github.com/RSC-Toolkit/Lung-SRAD)
- [BTS-CARD (ICASSP 2026)](https://arxiv.org/html/2510.22263) · [code](https://github.com/RSC-Toolkit/BTS-CARD)
- [Geometry-Aware / SAM-optimized AST (arXiv 2512.22564)](https://www.arxiv.org/pdf/2512.22564)
- [OPERA: Open Respiratory Acoustic Foundation Models (NeurIPS 2024 D&B)](https://arxiv.org/abs/2406.16148) · [code](https://github.com/evelyn0414/OPERA)
- [RSC-Toolkit GitHub organization](https://github.com/RSC-Toolkit)
- [Papers With Code — ICBHI leaderboard](https://paperswithcode.com/sota/audio-classification-on-icbhi-respiratory)
