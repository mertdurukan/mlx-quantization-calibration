"""Determinism + sampling contracts for src.benchmarks (SPEC §3, YAPILACAKLAR Görev 4)."""
import pytest

from src.benchmarks import load_items
import src.config as config


@pytest.mark.parametrize("benchmark", ["arc_challenge", "mmlu"])
def test_load_items_is_deterministic(benchmark):
    a = load_items(benchmark)
    b = load_items(benchmark)
    assert [x.item_id for x in a] == [x.item_id for x in b]


@pytest.mark.parametrize("benchmark", ["arc_challenge", "mmlu"])
def test_load_items_returns_n_items(benchmark):
    items = load_items(benchmark)
    assert len(items) == config.N_ITEMS


@pytest.mark.parametrize("benchmark", ["arc_challenge", "mmlu"])
def test_answer_idx_is_within_range(benchmark):
    items = load_items(benchmark)
    for item in items:
        assert 0 <= item.answer_idx < len(item.options)
        assert len(item.options) == len(item.labels)
        assert item.labels == [chr(ord("A") + i) for i in range(len(item.options))]


def test_mmlu_covers_all_57_subjects():
    items = load_items("mmlu")
    subjects = {item.item_id.rsplit("_", 1)[0][len("mmlu_"):] for item in items}
    assert len(subjects) == 57


def test_mmlu_item_ids_are_unique():
    items = load_items("mmlu")
    assert len({item.item_id for item in items}) == len(items)


def test_arc_challenge_sorted_by_id_ascending():
    items = load_items("arc_challenge")
    ids = [item.item_id for item in items]
    assert ids == sorted(ids)
