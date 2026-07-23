# Pre-Registration

## The Calibration Cost of the MLX Quantization Ladder

**Status:** PRE-REGISTERED. No experimental results exist at the time of this commit.
**Registered:** [DATE — set to commit date]
**Author:** Mert Durukan
**Repository rule:** This document is frozen at first commit. Any departure is recorded in
`DEVIATIONS.md` with an ISO timestamp, a reason, and whether it was decided BEFORE or AFTER
seeing the affected results. Deviations are reported in the write-up.

---

## 0. Pilot disclosure (read this first)

Before this pre-registration was written, a feasibility phase was run (commits `ffa07c7`,
`193c9d6`, `7c41237`, `fb4efda`). It established that the measurement is possible and fixed
the design space. Two things from it were known at freeze time and must be disclosed:

1. **A single-prompt observation.** On one prompt ("What is 2+2?") with Qwen2.5-0.5B, the
   4-bit model assigned *higher* probability to a wrong answer than bf16 (0.767 vs 0.687) and
   *lower* probability to the correct one (0.102 vs 0.197). This is N=1, one model, one item.
   It is **not a result**, but it was known before the hypotheses below were written and it
   informs their direction. Concealing it would weaken, not strengthen, this pre-registration.

2. **Accuracy measurements on 30 items of ARC-Challenge** for Qwen2.5 at 0.5B / 1.5B / 3B
   (33% / 73% / 73%). These informed the model-eligibility rule in section 4.2. No calibration
   metric was computed on any model at any point before freezing.

---

## 1. Background

Two literatures exist and do not touch.

**The MLX / quantization literature** measures throughput, latency, perplexity, and accuracy.
A survey of 2026 work — including multi-runtime Apple Silicon comparisons and community
quantization benchmarks — found none that reports a calibration metric.

**The LLM calibration literature** is large and active (RLHF-induced overconfidence,
multilingual calibration gaps, LLM-as-judge overconfidence, verbalized-confidence failures).
None of it examines post-training quantization as a factor.

The intersection is empty. Compressed models run on laptops and phones by default; if
compression silently makes a model *more confident while less correct*, that is a reliability
failure invisible to every metric currently reported.

A prior framing — that published studies disagreed on whether quantization damages calibration
— was checked in July 2026 and found largely **resolved**; that framing was abandoned rather
than pursued. This document targets the remaining gap: the MLX quantization ladder specifically,
on hardware where CUDA-only toolchains (GPTQ, AWQ, bitsandbytes) cannot run.

## 2. Research questions

- **RQ1.** How does probabilistic calibration change as MLX affine quantization bit-width
  decreases from bf16 through 8, 6, 5, 4, 3, to 2 bits?
- **RQ2.** Does quantization shift confidence *toward incorrect answers*, or does it degrade
  confidence symmetrically?
- **RQ3.** At matched bit-width and matched group size, does the quantization **mode**
  (`affine` vs `mxfp4`) change calibration?
- **RQ4 (secondary).** Do mixed-bit recipes (`mixed_2_6`, `mixed_3_4`, `mixed_3_6`,
  `mixed_4_6`) land between the calibration of their component uniform bit-widths?

**Terminology guard.** Throughout, *calibration* means **probabilistic calibration** — the
agreement between a model's stated confidence and its empirical correctness (ECE, calibration
slope/intercept). It never refers to the *calibration data* some quantization algorithms use
while compressing weights. The two are distinct and are never conflated in this repository.

## 3. Hypotheses (falsifiable, stated before any calibration measurement)

- **H1 — Monotone degradation.** ECE increases monotonically as bit-width decreases from 8 to
  2 bits.
  *Falsified if* the ECE ordering across {8, 6, 5, 4, 3, 2} is not monotone increasing, with
  95% intervals excluding the reversal being noise.

- **H2 — Directional overconfidence.** Quantization increases confidence assigned to incorrect
  answers relative to bf16, so the calibration intercept moves negative and mean confidence on
  incorrect predictions rises.
  *Falsified if* the intercept does not move away from 0 in the predicted direction, or if
  confidence on incorrect answers is statistically indistinguishable from bf16.

- **H3 — Mode matters at matched bits.** At 4 bits and group size 32, `mxfp4` and `affine`
  produce different ECE.
  *Falsified if* their ECE 95% intervals overlap across all models and benchmarks.

- **H4 (secondary) — Mixed-bit interpolation.** Each `mixed_a_b` recipe has ECE between the
  uniform `a`-bit and uniform `b`-bit conditions.
  *Falsified if* any recipe falls outside that range with a 95% interval excluding it.

**All outcomes will be published, including full or partial falsification.** A result of "the
calibration ladder is flat — quantization down to 4 bits is calibration-safe" would itself be
a finding and would be reported as prominently as the alternative.

## 4. Design (frozen)

### 4.1 Conditions (14 per model)

All conditions are produced **by us** from a single bf16 source per model, so that no
vendor-side variation between published checkpoints can act as a confounder.

| # | Condition | mode | bits | group_size |
|---|---|---|---|---|
| 1 | reference | — (bf16) | — | — |
| 2–7 | uniform ladder | affine | 8, 6, 5, 4, 3, 2 | 64 |
| 8 | group control (low) | affine | 4 | 32 |
| 9 | group control (high) | affine | 4 | 128 |
| 10 | mode contrast | mxfp4 | 4 | 32 |
| 11–14 | mixed recipes | `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6` | — | default |

**Condition 8 is load-bearing.** `mxfp4` requires group size 32 (verified: 16, 64, and 128 are
rejected with an explicit error). Comparing `mxfp4`(g=32) against `affine`(g=64) would confound
mode with group size. Condition 8 supplies the matched-group comparison that RQ3 requires.

`nf4` and `dynamic_quant` are **not available** in mlx-lm 0.31.3 (verified by execution) and
are therefore absent from this design.

**Effective bits per weight is recorded for every cell.** Nominal 4-bit at group size 64 costs
4.502 bits/weight in practice, because scales and biases are stored per group. Nominal
bit-width is a label, not a measurement.

### 4.2 Models — eligibility is mechanical and pre-specified

Candidate pool (bf16 sources, all verified accessible):
`Qwen2.5-1.5B-Instruct`, `Qwen2.5-3B-Instruct`, `Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`.

**Eligibility rule:** a model enters the main grid only if its **bf16 reference accuracy** on
the pre-registered sample is **>= 50%** on at least one benchmark.

Rationale, stated in advance: calibration asks whether a model is right when it is confident.
At accuracy near chance (25% for four options), confidence carries almost no information and
ECE is dominated by the base rate rather than by the quantization treatment. The threshold is
applied to the **bf16 reference only**, before any quantized cell is run, so it cannot be used
to select models on the basis of their quantized behaviour.

`Qwen2.5-0.5B` is **excluded from the main grid** on this rule (33% on 30 ARC-Challenge items
during feasibility, barely above chance). It is retained as a single documented **floor
control**: one full condition ladder is run on it and reported separately, to evidence the
claim that calibration cannot be meaningfully measured at that scale.

Models failing the eligibility rule are reported as excluded, with their accuracy — never
silently dropped.

### 4.3 Benchmarks — one protocol only

- **ARC-Challenge** (test split, n=1172) — first 1000 items by sorted `id`.
- **MMLU** (test split, n=14042) — 1000 items, stratified across all 57 subjects, seed 0.

Both are **letter-based multiple choice**, scored identically. HellaSwag and PIQA are
**excluded**: they require continuation scoring (length-normalized log-probability over full
endings), which is a different measurement protocol. Confidence calibration is known to be
highly sensitive to protocol, so mixing the two would confound quantization effects with
protocol effects. Continuation scoring is deferred to a separate study.

ARC-Easy is excluded from v1 to keep the grid tractable; noted as follow-up.

### 4.4 Measurement protocol (frozen)

- A **single prompt template**, committed to `prompts/` before any run and never varied:
  `Question: {question}\n{A) ...}\n{B) ...}\n...\nAnswer:`
- Confidence = softmax over the log-probabilities of the option-letter tokens (` A`, ` B`, ...)
  at the final position. Verbalized confidence is **not** used: it is independently broken
  (published ECE > 0.377) and would bury the quantization effect in protocol noise.
- Prediction = argmax over those option log-probabilities.
- **The first 20 items of every cell are warm-up and are discarded, not scored.** Verified
  necessity: an unwarmed 3B cell measured 693 ms/item versus 193 ms/item at steady state,
  because MLX loads weights lazily and compiles Metal kernels on first use. Without this rule,
  timing is meaningless and the first items are measured under a different execution regime.

### 4.5 Metrics

- **Calibration (the estimand):** ECE with **15 equal-mass bins**, calibration slope,
  calibration intercept, Brier score, mean confidence on correct vs incorrect predictions,
  overconfidence rate (fraction of high-confidence errors).
- **Reported but not the estimand:** accuracy, effective bits/weight, wall-clock per item.
- Equal-mass (quantile) bins are used, not equal-width: predicted probabilities concentrate,
  and equal-width bins leave most bins empty and understate calibration error.
- **Uncertainty:** percentile bootstrap over items, 2000 resamples, fixed seed. Every reported
  number carries a 95% interval. No point estimate is reported bare.
- **Primary contrasts are paired within item:** each quantized condition is compared to the
  bf16 reference on the *same* questions, differencing per item before summarising.

### 4.6 Confound guards

1. **Single source of weights.** Every condition for a model is produced from one bf16
   checkpoint by our own conversion. No mixing of vendor-published quantizations.
2. **Single machine, single runtime, single library version.** Apple M4 Pro, MLX only. Version
   pinned in `requirements.lock.txt`. **No cross-hardware or cross-runtime comparison in v1** —
   a different device is a confounder for numerical measurements of this kind.
3. **Matched group size for the mode contrast** (condition 8, section 4.1).
4. **Single prompt template** (section 4.4).
5. **Warm-up discarded** (section 4.4).
6. **Determinism contract.** A test must demonstrate that running the same cell twice produces
   **bit-identical** log-probabilities. If quantization or inference is non-deterministic, the
   entire comparison basis shifts and the design must be revised before any run. This is
   verified, not assumed.
7. **Full snapshot before conversion.** `snapshot_download` is called before every `convert`;
   mlx-lm's conversion path fails on incomplete Hugging Face caches (encountered and diagnosed
   during feasibility).

### 4.7 Compute, storage, and resumability

- Estimated: 4 models x 14 conditions x 2000 items ~ 112,000 forward passes.
  At 100–200 ms/item measured at steady state, **approximately 4–6 hours**, resumable.
- **Storage rule: convert -> evaluate -> delete.** Retaining every quantized checkpoint would
  exceed 100 GB. Only per-item log-probabilities are persisted (one file per cell), which
  allows every metric to be recomputed later without re-running a single model.
- Per-cell caching: a completed cell is never recomputed. The run must survive interruption.
- Failed cells are recorded with `status="failed"` and the exception text. **No cell is ever
  dropped, skipped, or silently retried.**

## 5. Analysis plan

- **Table 1 (H1):** ECE, slope, intercept, Brier per model x bit-width, paired against bf16,
  with 95% intervals. Monotonicity verdict.
- **Table 2 (H2):** mean confidence on correct vs incorrect, and overconfidence rate, per
  condition. Direction verdict.
- **Table 3 (H3):** `affine` vs `mxfp4` at 4 bits / group 32, per model x benchmark.
- **Table 4 (H4, secondary):** mixed recipes against their component uniform bit-widths.
- **Figure 1:** calibration curves, bf16 vs the bit ladder.
- **Figure 2:** ECE as a function of effective bits/weight (not nominal bits).
- **Figure 3:** confidence distribution on correct vs incorrect answers, bf16 vs 4-bit vs 2-bit.

Anything not listed above is **exploratory** and will be labelled as such in a separate section.
No hypothesis that is not stated in section 3 will be tested in the main tables.

## 6. Scope limits (stated in advance)

- Letter-based multiple-choice only. No claim about free-generation confidence, continuation
  scoring, or verbalized uncertainty.
- Models <= 3B parameters. No claim about 7B+ or frontier-scale behaviour.
- MLX on Apple Silicon only. **No claim** about GPTQ, AWQ, GGUF, or CUDA runtimes.
- Post-hoc recalibration (temperature scaling and similar) is **out of scope for v1** and is
  the natural follow-up.
- English-language benchmarks only.

## 7. Deviations

Logged in `DEVIATIONS.md`: what changed, why, whether decided BEFORE or AFTER seeing affected
results, and the impact on each hypothesis. `PREREG.md` itself is never edited after the first
commit.
