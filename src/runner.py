"""src/runner.py — grid loop, caching, convert->evaluate->delete (SPEC §3).

`cell_id`, `compute_eligibility`, and `assert_phase3_allowed` are implemented
ahead of Görev 7: they are pure gating/bookkeeping logic (no quantization
parameter is chosen or varied here), needed to make
tests/test_no_leakage.py's ordering contract testable before `run_all` exists
(SPEC §7 build order places test_no_leakage.py at step 6, run_all at step 7).
`run_all` itself is Görev 7's job.
"""
from pathlib import Path

import pandas as pd

from . import config


def cell_id(model_key: str, condition_tag: str, benchmark: str) -> str:
    """f'{model_key}__{condition_tag}__{benchmark}' — no hashing, human-readable."""
    return f"{model_key}__{condition_tag}__{benchmark}"


def compute_eligibility(cells: pd.DataFrame) -> dict:
    """PREREG §4.2 gate, computed from `condition == 'bf16'` rows ONLY.

    `cells` may already contain non-bf16 rows (in practice Phase 2 always runs
    before Phase 3 exists, but the filter is enforced here too as a
    structural guarantee — SPEC §0 prohibition 7 / PREREG §4.2: "cannot be
    used to select models on the basis of their quantized behaviour").

    Returns the results/eligibility.json record (SPEC §4): per-model,
    per-benchmark bf16 accuracy plus a mechanical `eligible` verdict
    (>= config.ELIGIBILITY_MIN_ACCURACY on at least one benchmark).
    Floor control models are additionally tagged `role: "floor_control"`.
    """
    bf16 = cells[cells["condition"] == "bf16"]
    out = {}
    for model_key, group in bf16.groupby("model"):
        record = {
            benchmark: float(bench_group["is_correct"].mean())
            for benchmark, bench_group in group.groupby("benchmark")
        }
        record["eligible"] = any(acc >= config.ELIGIBILITY_MIN_ACCURACY for acc in record.values())
        if model_key in config.FLOOR_CONTROL:
            record["role"] = "floor_control"
        out[model_key] = record
    return out


def assert_phase3_allowed(results_dir) -> None:
    """Raise unless results/eligibility.json exists (SPEC §0 prohibition 7,
    PREREG §4.2 ordering): quantized cells (Phase 3) must never run before
    the bf16 gate (Phase 2) has been evaluated and recorded."""
    path = Path(results_dir) / "eligibility.json"
    if not path.exists():
        raise RuntimeError(
            f"{path} does not exist — Phase 3 (quantized cells) cannot run before "
            "the bf16 eligibility gate (Phase 2) has been evaluated and recorded."
        )


def run_all(force: bool = False) -> None:
    """Executes the grid in the mandatory PHASE 1 -> PHASE 2 -> PHASE 3 order
    (SPEC §3). Not yet implemented — YAPILACAKLAR Görev 7."""
    raise NotImplementedError
