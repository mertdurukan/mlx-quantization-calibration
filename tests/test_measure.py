"""Regression test for the Llama-3.2 tokenizer BOS bug (YAPILACAKLAR AÇIK
BULGULAR, 2026-07-25, kritik).

`measure._score_item` looks up each option letter's logit index via
`tokenizer.encode(" " + label)`. Some tokenizers (e.g. Llama-3.2's) prepend a
leading special token (`<|begin_of_text|>`) to every `encode()` call, so the
label's own token is the LAST element of the returned list, not necessarily
the first. Reading `[0]` silently reads the constant BOS token for every
option on such tokenizers -- confirmed against the real Llama-3.2-1B
tokenizer, which returns `[128000, 362]` for `encode(" A")`.

This test exercises `measure._option_token_ids` against a fake tokenizer
that reproduces both behaviours (BOS-prepending and not), without loading a
real model.
"""
from src.measure import _option_token_ids


class _FakeTokenizerWithLeadingBOS:
    """Mimics Llama-3.2: every encode() call is prefixed with a constant
    BOS id, and the requested text's own token follows it."""

    BOS = 128000

    def encode(self, text):
        # deterministic per-label id, distinct from BOS and from each other
        return [self.BOS, 1000 + sum(map(ord, text))]


class _FakeTokenizerWithoutBOS:
    """Mimics Qwen: encode() returns only the requested text's token(s)."""

    def encode(self, text):
        return [1000 + sum(map(ord, text))]


def test_bos_prepending_tokenizer_yields_distinct_ids_per_label():
    tok = _FakeTokenizerWithLeadingBOS()
    ids = _option_token_ids(tok, ["A", "B", "C", "D"])

    assert tok.BOS not in ids  # the constant leading token must never be selected
    assert len(set(ids)) == 4  # each label maps to its own distinct token


def test_non_bos_tokenizer_still_works():
    tok = _FakeTokenizerWithoutBOS()
    ids = _option_token_ids(tok, ["A", "B"])

    expected = [tok.encode(" A")[0], tok.encode(" B")[0]]
    assert ids == expected
