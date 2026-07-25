"""Leakage/ordering contracts (SPEC §3, SPEC §7 build order item 6,
YAPILACAKLAR Görev 6):

1. measure.run_cell must not leak warmup items into scored output.
2. runner.compute_eligibility must compute accuracy from bf16 rows only, even
   when quantized rows for the same model are already present.
3. runner.assert_phase3_allowed must block Phase 3 until eligibility.json
   exists.

Mutation proof (PROTOKOL Kural 4) for all three was done interactively
against a deliberately broken implementation and is recorded in GECMIS.md —
the broken code is not committed here (SPEC §0 prohibition 2: no dead/skipped
code), per Kural 4's own instruction to delete it after showing the correct
test rejects it.
"""
import pandas as pd
import pytest

import src.config as config
from src.benchmarks import Item
from src.measure import run_cell
from src.runner import assert_phase3_allowed, compute_eligibility

MODEL_PATH = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"  # cached, no conversion needed


def _make_items(n):
    return [
        Item(item_id=f"t{i}", question=f"q{i}?", options=["foo", "bar"], labels=["A", "B"], answer_idx=i % 2)
        for i in range(n)
    ]


# --- warmup discard (measure.run_cell) ------------------------------------


@pytest.fixture(scope="module")
def scored():
    items = _make_items(config.N_WARMUP + 5)
    return items, run_cell(MODEL_PATH, items)


def test_scored_row_count_equals_input_items_not_more(scored):
    items, df = scored
    assert len(df) == len(items)  # warmup items are re-run, never appended as extra rows


def test_no_duplicate_item_ids_in_output(scored):
    _, df = scored
    assert len(df["item_id"]) == len(set(df["item_id"]))


def test_output_item_ids_match_input_exactly(scored):
    items, df = scored
    assert set(df["item_id"]) == {item.item_id for item in items}


# --- eligibility computed from bf16 rows only (runner.compute_eligibility) -


def test_eligibility_ignores_quantized_rows_for_the_same_model():
    # bf16: 3/10 correct = 0.3, below ELIGIBILITY_MIN_ACCURACY (0.50). A
    # QUANTIZED row for the same model claims 0.9 accuracy -- if eligibility
    # mixed it in, the verdict would flip to eligible=True. It must not.
    rows = (
        [{"model": "qwen2.5-1.5b", "condition": "bf16", "benchmark": "arc_challenge", "is_correct": c}
         for c in ([True] * 3 + [False] * 7)]
        + [{"model": "qwen2.5-1.5b", "condition": "affine_b4_g64", "benchmark": "arc_challenge", "is_correct": True}
           for _ in range(9)]
    )
    result = compute_eligibility(pd.DataFrame(rows))

    assert result["qwen2.5-1.5b"]["arc_challenge"] == pytest.approx(0.3)
    assert result["qwen2.5-1.5b"]["eligible"] is False


def test_eligibility_true_when_bf16_clears_bar_on_at_least_one_benchmark():
    rows = (
        [{"model": "qwen2.5-3b", "condition": "bf16", "benchmark": "arc_challenge", "is_correct": c}
         for c in ([True] * 7 + [False] * 3)]  # 0.7
        + [{"model": "qwen2.5-3b", "condition": "bf16", "benchmark": "mmlu", "is_correct": c}
           for c in ([True] * 4 + [False] * 6)]  # 0.4
    )
    result = compute_eligibility(pd.DataFrame(rows))

    assert result["qwen2.5-3b"]["eligible"] is True  # arc_challenge alone clears the bar


def test_floor_control_model_is_tagged_but_still_gets_a_mechanical_verdict():
    rows = [{"model": "qwen2.5-0.5b", "condition": "bf16", "benchmark": "arc_challenge", "is_correct": c}
            for c in ([True] * 3 + [False] * 7)]  # 0.3
    result = compute_eligibility(pd.DataFrame(rows))

    assert result["qwen2.5-0.5b"]["role"] == "floor_control"
    assert result["qwen2.5-0.5b"]["eligible"] is False


# --- Phase 3 ordering gate (runner.assert_phase3_allowed) ------------------


def test_phase3_blocked_without_eligibility_file(tmp_path):
    with pytest.raises(RuntimeError):
        assert_phase3_allowed(tmp_path)


def test_phase3_allowed_once_eligibility_file_exists(tmp_path):
    (tmp_path / "eligibility.json").write_text("{}")
    assert_phase3_allowed(tmp_path)  # must not raise
