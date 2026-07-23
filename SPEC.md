# SPEC — Implementation Contract

`PREREG.md` defines **what** the experiment is and is frozen.
This document defines **how the code is shaped**. It may be updated when it drifts from
reality, with a Changelog entry — but never in a way that changes the science.

---

## 0. PROHIBITIONS (read before writing any line)

1. **NEVER change a quantization parameter to make a cell work.** The 14 conditions in
   PREREG §4.1 are frozen. If a condition fails, record it as `status="failed"` with the
   exception text. Do not substitute a different bit-width or group size.

2. **NEVER skip, drop, or silently retry a cell.** A failing cell is data. No
   `try/except: continue`. No filtering out models or benchmarks that look inconvenient.

3. **NEVER vary the prompt template.** One template, committed to `prompts/` before the
   first run. Confidence calibration is highly protocol-sensitive; a second template is a
   second experiment.

4. **NEVER edit `PREREG.md`.** Append to `DEVIATIONS.md` instead, with an ISO timestamp,
   a reason, and whether the decision was made BEFORE or AFTER seeing affected results.

5. **NEVER report a number without a 95% interval.**

6. **NEVER hand-edit anything under `results/`.** It is generated. Fix the code, regenerate.

7. **NEVER run a quantized cell for a model before its bf16 reference cell has run and the
   eligibility rule (PREREG §4.2) has been evaluated.** Ordering is part of the design: the
   50% threshold must be applied to bf16 only, before any quantized behaviour is observed.

8. **NEVER add a dependency** without pinning it in `requirements.txt` and regenerating
   `requirements.lock.txt` in the same commit.

9. **NEVER rewrite git history.** The audit trail is the artifact.

10. **When unsure, stop and ask.** A wrong guess contaminates a scientific result, which is
    worse than a delay.

---

## 1. Repository layout (create exactly this)

```
.
├── PREREG.md                  # frozen
├── SPEC.md                    # this file
├── DEVIATIONS.md              # append-only
├── PROTOKOL.md                # operating discipline
├── requirements.txt
├── requirements.lock.txt
├── Makefile
├── prompts/
│   └── mc_letter.txt          # THE prompt template, frozen before first run
├── src/
│   ├── __init__.py
│   ├── config.py              # all constants; no magic numbers elsewhere
│   ├── quantize.py            # condition -> quantized model on disk
│   ├── benchmarks.py          # load + sample + normalize ARC-C / MMLU
│   ├── measure.py             # model + items -> per-item log-probs
│   ├── metrics.py             # ECE, slope, intercept, Brier, overconf, bootstrap
│   ├── runner.py              # grid loop, caching, convert->evaluate->delete
│   └── analyze.py             # tables + figures
├── tests/
│   ├── test_determinism.py    # PREREG §4.6.6 contract (already passing)
│   ├── test_metrics.py        # known-answer tests, written BEFORE metrics.py
│   ├── test_prompt_frozen.py  # template hash must match a committed constant
│   └── test_no_leakage.py     # eligibility ordering + warmup discard contracts
├── results/
│   ├── cells/                 # one parquet per cell: PER-ITEM rows
│   ├── meta/                  # one json per cell: status, effective bits, timing
│   ├── tables/
│   └── figures/
└── scratch/                   # throwaway; gitignored
```

Feasibility scripts from the discovery phase (`feasibility_test.py`, `discover*.py`,
`verify*.py`, `scaling.py`, `thermal.py`, `timing.py`, `conv_test.py`, `bench*.py`) stay in
the repo as the provenance of the design. Move them to `scratch/` only if they are also
committed there — do not delete them.

---

## 2. `src/config.py` — frozen constants

```python
from typing import Final

SEED: Final[int] = 0

# --- models (PREREG §4.2) ---
MAIN_MODELS: Final[dict[str, str]] = {
    "qwen2.5-1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-bf16",
    "qwen2.5-3b":   "mlx-community/Qwen2.5-3B-Instruct-bf16",
    "llama3.2-1b":  "mlx-community/Llama-3.2-1B-Instruct-bf16",
    "llama3.2-3b":  "mlx-community/Llama-3.2-3B-Instruct-bf16",
}
FLOOR_CONTROL: Final[dict[str, str]] = {
    "qwen2.5-0.5b": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
}
ELIGIBILITY_MIN_ACCURACY: Final[float] = 0.50   # bf16 reference only

# --- conditions (PREREG §4.1) --- (tag, mode, bits, group_size)
CONDITIONS: Final[list[tuple[str, str | None, int | None, int | None]]] = [
    ("bf16",           None,     None, None),
    ("affine_b8_g64",  "affine", 8,    64),
    ("affine_b6_g64",  "affine", 6,    64),
    ("affine_b5_g64",  "affine", 5,    64),
    ("affine_b4_g64",  "affine", 4,    64),
    ("affine_b3_g64",  "affine", 3,    64),
    ("affine_b2_g64",  "affine", 2,    64),
    ("affine_b4_g32",  "affine", 4,    32),   # load-bearing: matched group for mxfp4
    ("affine_b4_g128", "affine", 4,    128),
    ("mxfp4_b4_g32",   "mxfp4",  4,    32),
    ("mixed_2_6",      "recipe", None, None),
    ("mixed_3_4",      "recipe", None, None),
    ("mixed_3_6",      "recipe", None, None),
    ("mixed_4_6",      "recipe", None, None),
]

# --- benchmarks (PREREG §4.3) ---
N_ITEMS: Final[int] = 1000
BENCHMARKS: Final[list[str]] = ["arc_challenge", "mmlu"]

# --- measurement (PREREG §4.4) ---
N_WARMUP: Final[int] = 20        # discarded, never scored
PROMPT_FILE: Final[str] = "prompts/mc_letter.txt"

# --- metrics (PREREG §4.5) ---
ECE_N_BINS: Final[int] = 15      # equal-MASS
BOOTSTRAP_N: Final[int] = 2_000
CI_LOW: Final[float] = 2.5
CI_HIGH: Final[float] = 97.5
OVERCONF_THRESHOLD: Final[float] = 0.90   # "high-confidence error" cutoff
```

**Rule:** every number used anywhere in `src/` comes from here. A numeric literal in
`runner.py` or `measure.py` is a bug.

---

## 3. Module contracts (exact signatures)

### `src/benchmarks.py`
```python
@dataclass(frozen=True)
class Item:
    item_id: str
    question: str
    options: list[str]        # option TEXTS, in order
    labels: list[str]         # option LETTERS, e.g. ["A","B","C","D"]
    answer_idx: int           # index into options/labels

def load_items(benchmark: str) -> list[Item]:
    """Deterministic, seeded sample of exactly N_ITEMS.

    arc_challenge: first N_ITEMS of the test split sorted by `id` ascending.
    mmlu:          N_ITEMS from the test split, stratified across all 57 subjects,
                   seeded with config.SEED. Stratification: proportional allocation,
                   remainder assigned by subject name ascending. Deterministic.

    MMLU has `choices` as a plain list and `answer` as an int; ARC has a dict with
    `text`/`label` and `answerKey` as a letter. Both normalize to Item. Letters are
    always regenerated as A, B, C, ... from position, never taken from the source, so
    the two benchmarks are scored identically.
    """
```

### `src/quantize.py`
```python
def build(model_key: str, condition_tag: str, out_dir: str) -> dict:
    """Produce one quantized model on disk from the single bf16 source.

    MUST call snapshot_download(src) before convert() — mlx-lm's conversion path fails on
    incomplete HF caches (PREREG §4.6.7).

    bf16 condition: no quantization; returns the resolved snapshot path.
    affine/mxfp4:   convert(..., quantize=True, q_bits=, q_group_size=, q_mode=)
    recipe:         convert(..., quant_predicate=<recipe name from QUANT_RECIPES>)

    Returns {"path", "effective_bits", "size_mb", "convert_seconds"}.
    effective_bits is PARSED FROM THE CONVERSION LOG, not assumed: nominal 4-bit is
    4.501-4.502 bits/weight in practice (PREREG §4.1).
    """

def teardown(path: str) -> None:
    """Delete a converted model directory. Called after every non-bf16 cell
    (PREREG §4.7 storage rule). Never deletes the HF cache."""
```

### `src/measure.py`
```python
def run_cell(model_path: str, items: list[Item]) -> pd.DataFrame:
    """Score items with ONE model. Returns one row PER ITEM (schema in §4).

    - Loads the frozen template from config.PROMPT_FILE.
    - The FIRST config.N_WARMUP items are run and DISCARDED (PREREG §4.4). They are not
      scored, not returned, not counted. Warmup uses the same items each time so it is
      deterministic.
    - Confidence = softmax over the option-letter token log-probs at the final position.
      Letter tokens are encoded as " A", " B", ... (leading space).
    - Never raises on a single item: an item that fails is returned with status="failed"
      and the exception text, and the cell continues.
    """
```

### `src/metrics.py`
Same family as the sibling ML study. Every function `(y_true, y_prob, **kw) -> float`.

- `ece(y_correct, y_conf)` — **15 equal-MASS bins** via `np.array_split` on confidence-sorted
  order. Not equal-width.
- `cal_slope` / `cal_intercept` — logistic regression of correctness on `logit(confidence)`;
  intercept uses `statsmodels` GLM with `offset=`. Clip confidence to `[1e-6, 1-1e-6]` first.
- `brier(y_correct, y_conf)`
- `overconfidence_rate(y_correct, y_conf, threshold=OVERCONF_THRESHOLD)` — fraction of items
  with confidence above threshold that are wrong.
- `mean_conf_by_correctness(y_correct, y_conf) -> tuple[float, float]`
- `bootstrap_ci(y_correct, y_conf, metric_fn, n=BOOTSTRAP_N, seed=SEED, **metric_kwargs)`
  — percentile bootstrap over ITEMS. **Point estimate is the metric on the FULL sample**,
  not the bootstrap mean. `**metric_kwargs` MUST be forwarded to both the point estimate and
  every resample.

> The sibling study shipped a `bootstrap_ci` that silently dropped kwargs and would have
> crashed at analysis time. That is why the forwarding requirement is written here.

### `src/runner.py`
```python
def cell_id(model_key: str, condition_tag: str, benchmark: str) -> str:
    """f'{model_key}__{condition_tag}__{benchmark}' — no hashing, human-readable."""

def run_all(force: bool = False) -> None:
    """Executes the grid in a MANDATORY ORDER:

      PHASE 1 — all bf16 reference cells, every model x benchmark.
      PHASE 2 — evaluate the eligibility rule (PREREG §4.2) from PHASE 1 results ONLY.
                Write results/eligibility.json recording each model's bf16 accuracy and
                its verdict. This file is the audit record of the gate.
      PHASE 3 — all quantized cells, for eligible models + the floor control.

    Phase 2 MUST NOT be influenced by any quantized result, because none exists yet.
    That ordering is the whole point of the rule.

    Caching: a cell whose results/cells/{cell_id}.parquet exists is skipped unless force.
    Storage: after each non-bf16 cell, quantize.teardown() removes the converted model.
    Failure: never raises, never skips; failures are written with status='failed'.
    """
```

---

## 4. Output schema (FROZEN)

### `results/cells/{cell_id}.parquet` — one row per scored item

| column | type | notes |
|---|---|---|
| `cell_id` | str | |
| `model` | str | key from MAIN_MODELS / FLOOR_CONTROL |
| `condition` | str | tag from CONDITIONS |
| `benchmark` | str | |
| `item_id` | str | from Item |
| `n_options` | int | |
| `answer_idx` | int | ground truth |
| `pred_idx` | int | argmax over option log-probs |
| `is_correct` | bool | |
| `conf_pred` | float | softmax prob of the PREDICTED option |
| `conf_true` | float | softmax prob of the CORRECT option |
| `logprobs` | list[float] | raw per-option log-probs — **mandatory** |
| `latency_ms` | float | |
| `status` | str | `"ok"` or `"failed"` |
| `error` | str \| null | |

**`logprobs` is the single most important column.** Storing raw per-option log-probs means
every metric can be recomputed later without re-running a single model. The sibling study's
equivalent (`y_prob_path`) is what made its whole analysis phase cheap.

### `results/meta/{cell_id}.json` — one object per cell

```json
{"cell_id": "...", "model": "...", "condition": "...", "benchmark": "...",
 "effective_bits": 4.501, "size_mb": 980.2, "convert_seconds": 3.4,
 "n_items_scored": 1000, "n_warmup_discarded": 20,
 "n_failed_items": 0, "wall_seconds": 132.7,
 "mlx_version": "...", "mlx_lm_version": "...", "status": "ok"}
```

### `results/eligibility.json` — the gate record
```json
{"qwen2.5-1.5b": {"arc_challenge": 0.73, "mmlu": 0.61, "eligible": true},
 "qwen2.5-0.5b": {"arc_challenge": 0.33, "mmlu": 0.28, "eligible": false,
                  "role": "floor_control"}}
```

---

## 5. Caching, resumability, storage

- `run_cell` result is written immediately after computation; the parquet is the completion
  marker.
- `run_all` checks existence and skips. A 140-cell run **will** be interrupted; resuming must
  not recompute finished cells.
- After each non-bf16 cell, the converted model directory is deleted. Only the HF cache of
  bf16 sources persists (roughly 20 GB across five models).
- Peak disk at any moment: one bf16 source plus one quantized copy.

---

## 6. Make targets

| target | does |
|---|---|
| `make setup` | `python3 -m venv .venv && pip install -r requirements.txt` |
| `make verify` | functional smoke test: convert a tiny model, load it, read one log-prob, compute one ECE. **Imports alone do not count** — the sibling study had a green import check over a broken environment. |
| `make test` | `pytest tests/ -q` |
| `make prompt-freeze` | prints the SHA-256 of `prompts/mc_letter.txt`; must match `tests/test_prompt_frozen.py` |
| `make pilot` | one model x three conditions (bf16, affine_b4_g64, affine_b2_g64) x arc_challenge |
| `make reproduce` | `make test` then `python -m src.runner` then `python -m src.analyze` |

All targets invoke `./.venv/bin/python` explicitly; no manual activation required.

---

## 7. Build order (tests before implementation)

1. `config.py` + `prompts/mc_letter.txt` + `test_prompt_frozen.py`
2. `tests/test_metrics.py` — **known-answer tests, reviewed before `metrics.py` exists**
3. `src/metrics.py` — implement to pass them; **no test may be weakened to make it pass**
4. `src/benchmarks.py` + a determinism test (same sample twice = identical item ids)
5. `src/quantize.py` + `src/measure.py`
6. `tests/test_no_leakage.py` — asserts warmup items are excluded from output, and that
   Phase 3 cannot run before `eligibility.json` exists. **Prove it can fail** (Rule 4 of
   PROTOKOL.md): write a deliberately broken runner that scores warmup items, show the test
   rejects it, delete the broken version.
7. `src/runner.py`
8. `make pilot` — hand-check before scaling
9. Full run
10. `src/analyze.py` — only the four pre-registered tables and three figures

---

## 8. Pilot acceptance (do this by hand, do not delegate)

After `make pilot`:

- Exactly 3 cells, `status="ok"`, 1000 scored items each, 20 warmup discarded each.
- `effective_bits` present and **not** equal to nominal (expect ~4.50 for b4_g64).
- **Sanity check with a predicted direction:** the 2-bit cell should show clearly worse
  accuracy AND worse ECE than bf16. If 2-bit looks *identical* to bf16, quantization is not
  actually being applied — check that `Linear` became `QuantizedLinear`, which was a real
  no-op trap during feasibility.
- Time one cell, multiply by 140. **If the projection exceeds 8 hours, stop and report before
  scaling up.**

---

## 9. Changelog

- 2026-07-24 — initial contract, written after the feasibility gate and the frozen PREREG.
