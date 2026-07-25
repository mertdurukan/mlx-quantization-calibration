"""src/runner.py — grid loop, caching, convert->evaluate->delete (SPEC §3)."""
import json
import os
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
import pandas as pd

from . import benchmarks, config, measure, quantize

_CELL_COLUMNS = [
    "cell_id", "model", "condition", "benchmark", "item_id", "n_options",
    "answer_idx", "pred_idx", "is_correct", "conf_pred", "conf_true",
    "logprobs", "latency_ms", "status", "error",
]


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


def _run_and_write_cell(model_key: str, condition_tag: str, benchmark: str, results_dir: Path, force: bool) -> None:
    """Run one cell (build -> measure -> write parquet+meta -> teardown).

    Never raises (SPEC §3): any exception during build/measure is caught and
    recorded as a `status="failed"` meta record with the exception text; the
    per-item failure path (a single item erroring inside measure.run_cell) is
    handled separately, inside measure.py, and does not reach here.
    """
    cid = cell_id(model_key, condition_tag, benchmark)
    cell_path = results_dir / "cells" / f"{cid}.parquet"
    meta_path = results_dir / "meta" / f"{cid}.json"
    if cell_path.exists() and not force:
        return

    t0 = time.perf_counter()
    if condition_tag != "bf16":
        out_dir = tempfile.mkdtemp(prefix=f"{cid}__")
        os.rmdir(out_dir)  # mlx_lm.convert() refuses to write to a path that already exists
    else:
        out_dir = ""
    meta = {
        "cell_id": cid, "model": model_key, "condition": condition_tag, "benchmark": benchmark,
        "mlx_version": mx.__version__, "mlx_lm_version": mlx_lm.__version__,
    }
    try:
        build_info = quantize.build(model_key, condition_tag, out_dir)
        items = benchmarks.load_items(benchmark)
        df = measure.run_cell(build_info["path"], items)

        df["cell_id"] = cid
        df["model"] = model_key
        df["condition"] = condition_tag
        df["benchmark"] = benchmark
        df = df[_CELL_COLUMNS]
        df.to_parquet(cell_path, index=False)

        meta.update({
            "effective_bits": build_info["effective_bits"],
            "size_mb": build_info["size_mb"],
            "convert_seconds": build_info["convert_seconds"],
            "n_items_scored": len(df),
            "n_warmup_discarded": config.N_WARMUP,
            "n_failed_items": int((df["status"] == "failed").sum()),
            "wall_seconds": time.perf_counter() - t0,
            "status": "ok",
        })
    except Exception as exc:
        meta.update({
            "effective_bits": None, "size_mb": None, "convert_seconds": None,
            "n_items_scored": 0, "n_warmup_discarded": config.N_WARMUP,
            "n_failed_items": None, "wall_seconds": time.perf_counter() - t0,
            "status": "failed", "error": str(exc),
        })
    finally:
        if condition_tag != "bf16":
            try:
                quantize.teardown(out_dir)
            except Exception:
                pass  # cleanup is best-effort; never masks the cell's own outcome

    meta_path.write_text(json.dumps(meta, indent=2))


def run_all(force: bool = False) -> None:
    """Executes the grid in a MANDATORY ORDER (SPEC §3):

      PHASE 1 — all bf16 reference cells, every model x benchmark.
      PHASE 2 — evaluate the eligibility rule from PHASE 1 results ONLY, write
                results/eligibility.json.
      PHASE 3 — all quantized cells, for eligible models + the floor control
                (PREREG §4.2: floor control runs the full condition ladder
                regardless of its own eligibility verdict).
    """
    results_dir = Path("results")
    (results_dir / "cells").mkdir(parents=True, exist_ok=True)
    (results_dir / "meta").mkdir(parents=True, exist_ok=True)

    all_models = {**config.MAIN_MODELS, **config.FLOOR_CONTROL}

    # PHASE 1
    for model_key in all_models:
        for benchmark in config.BENCHMARKS:
            _run_and_write_cell(model_key, "bf16", benchmark, results_dir, force)

    # PHASE 2 — bf16 rows only, read back from disk (not carried in memory from
    # Phase 1) so a resumed run re-derives eligibility from whatever bf16 cells
    # actually exist on disk, not from an in-process cache.
    bf16_frames = []
    for model_key in all_models:
        for benchmark in config.BENCHMARKS:
            cid = cell_id(model_key, "bf16", benchmark)
            path = results_dir / "cells" / f"{cid}.parquet"
            if path.exists():
                bf16_frames.append(pd.read_parquet(path))
    combined = pd.concat(bf16_frames, ignore_index=True) if bf16_frames else pd.DataFrame(
        columns=["model", "condition", "benchmark", "is_correct"]
    )
    eligibility = compute_eligibility(combined)
    (results_dir / "eligibility.json").write_text(json.dumps(eligibility, indent=2))

    # PHASE 3
    assert_phase3_allowed(results_dir)
    eligible_models = [
        model_key for model_key, record in eligibility.items()
        if record.get("eligible") or record.get("role") == "floor_control"
    ]
    for model_key in eligible_models:
        for condition_tag, _mode, _bits, _group_size in config.CONDITIONS:
            if condition_tag == "bf16":
                continue
            for benchmark in config.BENCHMARKS:
                _run_and_write_cell(model_key, condition_tag, benchmark, results_dir, force)


def run_pilot() -> None:
    """SPEC §8: one model x three conditions (bf16, affine_b4_g64,
    affine_b2_g64) x arc_challenge, hand-checked before scaling to the full
    grid. Not part of SPEC §3's `run_all` contract, so it does not touch
    `run_all`'s signature; it reuses the same mandatory phase order (a pilot
    quantized cell must not run before its bf16 reference + eligibility
    check, same as the full grid) via the same building blocks.

    Pilot cells land in the normal results/ tree under their normal cell_id,
    so the full run's Phase 1 caching (SPEC §5) skips and reuses them rather
    than recomputing -- and the full run's own Phase 2 recomputes
    eligibility.json from ALL bf16 cells once it runs, so this call's
    single-model eligibility.json is a scratch artifact, not a lasting one.
    """
    results_dir = Path("results")
    (results_dir / "cells").mkdir(parents=True, exist_ok=True)
    (results_dir / "meta").mkdir(parents=True, exist_ok=True)

    model_key = config.PILOT_MODEL
    benchmark = config.PILOT_BENCHMARK

    _run_and_write_cell(model_key, "bf16", benchmark, results_dir, force=False)

    cid = cell_id(model_key, "bf16", benchmark)
    path = results_dir / "cells" / f"{cid}.parquet"
    combined = pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=["model", "condition", "benchmark", "is_correct"]
    )
    eligibility = compute_eligibility(combined)
    (results_dir / "eligibility.json").write_text(json.dumps(eligibility, indent=2))

    assert_phase3_allowed(results_dir)
    for condition_tag in config.PILOT_CONDITIONS:
        _run_and_write_cell(model_key, condition_tag, benchmark, results_dir, force=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="SPEC §8 pilot grid instead of the full run")
    parser.add_argument("--force", action="store_true", help="recompute cells even if cached")
    args = parser.parse_args()

    if args.pilot:
        run_pilot()
    else:
        run_all(force=args.force)
