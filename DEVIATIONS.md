# DEVIATIONS LOG

Append-only. Every departure from `PREREG.md` is recorded here with an ISO-8601 timestamp,
what changed, why, and whether it was decided BEFORE or AFTER seeing the affected results.

The "before/after" field is the important one: a deviation decided after seeing results is a
potential source of bias and must be reported as such in the paper.

`PREREG.md` itself is never edited. This file is the only place a departure may live.

Format:

```
## YYYY-MM-DDTHH:MM:SSZ — <short title>
- **Changed:** what, precisely
- **Reason:** why it was unavoidable
- **Decided:** BEFORE seeing affected results / AFTER seeing affected results
- **Impact on hypotheses:** none / H1 / H2 / H3 / H4
```

---

> All six entries below were found by an **independent zero-context audit** on 2026-07-26
> (PROTOKOL Kural 10), not by the authoring context, and each was independently re-verified
> before being logged. Until that audit this file said "no deviations yet" — it was empty
> because the departures had not been *recognised*, not because there were none. That is the
> single most important thing this log now records.

## 2026-07-26T00:00:00Z — H1's ordering criterion was implemented over adjacent bit-widths only

- **Changed:** `src/analyze.py::_h1_ladder_verdict` tested only *adjacent* pairs on the
  bit ladder (`zip(rows, rows[1:])`). `PREREG.md` H1 states the criterion as "the ECE
  **ordering across {8, 6, 5, 4, 3, 2}**", and an ordering over a set is a claim about every
  pair: a reversal spread across several steps, each step individually CI-overlapping, breaks
  the ordering while being invisible to an adjacent-only test. **Corrected** to test all pairs,
  which is what was pre-registered. Two additional CI-confirmed reversals surfaced:
  `qwen2.5-1.5b/mmlu` 8→3 bits and `qwen2.5-3b/mmlu` 8→4 bits.
- **Reason:** implementation error, not a design choice. The narrower rule was strictly weaker
  than the pre-registered one and biased toward PASS, i.e. toward *not* falsifying the study's
  own hypothesis.
- **Decided:** AFTER seeing affected results — unavoidably so, since the discrepancy was found
  by auditing the finished analysis. The correction was chosen without reference to which
  direction it would move the conclusions; it moves them **against** the paper's original
  framing (more instability, in two models rather than one), which is the opposite of a
  favourable bias.
- **Impact on hypotheses:** H1. The overall verdict was FAIL before and remains FAIL; the
  per-cell count changes and the "localized to one model" reading in the original draft does
  not survive.

## 2026-07-26T00:00:01Z — H1's reversal test uses unpaired marginal intervals

- **Changed:** nothing (disclosed, not corrected). `_h1_ladder_verdict` decides whether a
  reversal is noise by asking whether two **marginal** ECE bootstrap intervals overlap.
  `PREREG.md` §"Primary contrasts are paired within item" mandates paired contrasts, and
  `_paired_bootstrap_delta` — already implemented and used for H2/H3/H4 — would give the
  paired test. Overlap of marginal intervals is markedly more conservative: an independent
  re-test with paired deltas finds further significant reversals
  (`qwen2.5-1.5b/mmlu` 4→3 bits, `qwen2.5-3b/mmlu` 5→4 bits).
- **Reason:** the implemented test matches `SPEC.md`'s wording, so this is a
  specification-level divergence from `PREREG.md` rather than a coding slip. Switching H1 to
  paired deltas between every bit-width pair is a substantive change to the analysis that
  should be pre-registered in its own right, not retrofitted after seeing results.
- **Decided:** AFTER seeing affected results — hence disclosed rather than changed. The
  direction of the bias is stated explicitly in the paper: the criterion actually used
  **under-detects** falsification.
- **Impact on hypotheses:** H1 (per-cell verdicts are criterion-dependent; the overall FAIL is
  not).

## 2026-07-26T00:00:02Z — The floor-control model satisfied the eligibility rule as written

- **Changed:** `PREREG.md` sets the eligibility rule as "bf16 reference accuracy **on the
  pre-registered sample** is >= 50% on at least one benchmark", and pre-commits separately to
  treating `qwen2.5-0.5b` as floor control on the basis of 33% accuracy over 30 ARC items
  during feasibility. Measured on the pre-registered 1000-item sample, `qwen2.5-0.5b` scores
  **0.518** on ARC-Challenge and therefore **passes** the rule as written
  (`results/eligibility.json` records `"eligible": true`). It is excluded from the main grid by
  a `role == "floor_control"` field in `src/analyze.py`, not by the rule.
- **Reason:** `PREREG.md` contains two mutually inconsistent commitments — the rule's own
  threshold applied to the pre-registered sample, versus the named exclusion justified by a
  30-item pilot. The named exclusion was honoured because it is the more specific pre-commitment
  and because promoting the model post hoc would mean re-deciding the model pool after seeing
  quantized results. `PREREG.md` is frozen and was not edited.
- **Decided:** the exclusion itself was decided BEFORE any result (it is in the frozen
  pre-registration). The *recognition* that the rule as written admits the model came AFTER.
- **Impact on hypotheses:** none on H1–H4 (the model was never in the main grid). It does
  weaken one interpretive claim: the floor control cannot evidence that "calibration is not
  meaningfully measurable near chance accuracy", because at 0.518 it is not near chance. The
  paper states this instead of the original claim.

## 2026-07-26T00:00:03Z — Warm-up items are re-scored rather than discarded

- **Changed:** `PREREG.md` states "the first 20 items of every cell are warm-up and are
  **discarded, not scored**", which implies 980 scored items per cell. `src/measure.py` runs the
  first 20 items once and throws the result away, then scores **all 1000 items, including those
  20**, in a second pass. Every cell parquet has 1000 rows and all 20 warm-up `item_id`s are
  present in the scored output (verified).
- **Reason:** `PREREG.md` justifies the warm-up rule solely by measurement validity ("without
  this rule, timing is meaningless"). The implementation satisfies that rationale more fully
  than the literal rule does — every scored item is measured in the warmed steady state, rather
  than 980 warmed items plus 20 discarded ones — so the implementation was kept and the
  documents corrected. `SPEC.md` and `paper.md` both described the literal rule and were wrong
  about what the code does.
- **Decided:** AFTER seeing affected results. Re-running the 7-hour grid to discard 20 of 1000
  items per cell would change each ECE by roughly the weight of 2% of the sample while removing
  warmed measurements; the honest disclosure is preferred to a costly re-run that makes the
  measurement slightly worse.
- **Impact on hypotheses:** none in direction or significance — the 20 items are a fixed 2% of
  every cell and identical across all conditions, so paired within-item contrasts are unaffected.

## 2026-07-26T00:00:04Z — Two estimand definitions left open by the pre-registration were resolved in code

- **Changed:** (a) `PREREG.md` defines the overconfidence rate only as "fraction of
  high-confidence errors", with no threshold; `src/config.py` fixes it at >90% confidence, and
  strictly above rather than at. (b) `PREREG.md` does not say whether H4's "component range" is
  bounded by the components' ECE point estimates or by their intervals; the code uses point
  estimates.
- **Reason:** both are ambiguities in the frozen text that had to be resolved to compute
  anything. Both resolutions are the stricter/more conservative reading: a 90% threshold with a
  strict inequality, and point-estimate bounds (narrower than interval bounds, so easier to
  fall outside and therefore easier to falsify H4).
- **Decided:** the code was written BEFORE results were seen, so the resolutions themselves are
  not post-hoc; the *disclosure* is (they were not logged at the time).
- **Impact on hypotheses:** H2 (threshold choice), H4 (range definition). Neither verdict is
  known to be sensitive to the alternative reading, and no sensitivity analysis was run —
  stated here rather than implied.

## 2026-07-26T00:00:05Z — Tables 1 and 2 report a subset of their pre-registered contents

- **Changed:** `PREREG.md` §"Table 1 (H1)" specifies "ECE, slope, intercept, Brier per model x
  bit-width, **paired against bf16**, with 95% intervals". The paper's Table 1 reports ECE only,
  and reports raw per-cell ECE rather than the bf16-paired deltas. `PREREG.md` §"Table 2 (H2)"
  specifies "mean confidence on correct vs incorrect, and **overconfidence rate**, per
  condition"; the paper's Table 2 reports Δintercept and Δconf(incorrect), and neither mean
  confidence on correct nor the overconfidence rate appears anywhere in the paper.
- **Reason:** presentation, not analysis. All of it was computed and is published:
  `table1_h1_bit_ladder.csv` carries `slope`, `intercept`, `brier` and every `delta_*` column
  with intervals, and `table2_h2_confidence_direction.csv` carries `mean_conf_correct`,
  `overconfidence_rate` and its interval. The full pre-registered Table 1 is 6 bit-widths x 6
  model-benchmark cells x 4 estimands x (point, lo, hi) and does not fit a single-column paper
  page legibly, so ECE — the estimand H1 is stated in terms of — was shown and the rest left in
  the CSV. The omission was not deliberate concealment, but it was also not disclosed, and the
  captions did not point readers to the CSV for the remainder.
- **Decided:** AFTER seeing results (the tables were laid out during the write-up). Note the
  direction: the omitted estimands are the ones that would let a reader check H1 by a route other
  than ECE. Nothing was added that PREREG does not authorise — an independent audit checked
  specifically for unregistered estimands smuggled into the main tables and found none. The
  defect is one-directional: omission.
- **Impact on hypotheses:** none on any verdict (all four are computed from the full data, not
  from what the paper prints). It does weaken reader verifiability, which is the point of
  pre-registering table contents. §4.1 and §4.2 now state the omission and name the columns.
