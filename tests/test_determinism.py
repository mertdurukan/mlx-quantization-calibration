"""PREREG §4.6.6 — determinism contract + mutation proof.

Rewritten from a print-based script into real pytest assertions (AÇIK BULGULAR,
2026-07-25): the original had no `assert`, so pytest collected it, ran the full
conversion at import time on every `make test`, and reported nothing either way.
The contract itself (same cell run twice -> bit-identical log-probs, and injected
noise IS caught) was already verified passing before this rewrite — this only
changes how that verification is expressed and re-checked.
"""
import shutil

import mlx.core as mx
import pytest
from datasets import load_dataset
from huggingface_hub import snapshot_download
from mlx_lm import convert, load

NL = chr(10)
SRC = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"
N = 20
OUT = "/tmp/mlxq_det"


def _logprobs(model, tokenizer, items, n=N, noise=0.0):
    out = []
    for i in range(n):
        ex = items[i]
        labs = ex["choices"]["label"]
        opts = NL.join(l + ") " + t for l, t in zip(labs, ex["choices"]["text"]))
        ids = tokenizer.encode("Question: " + ex["question"] + NL + opts + NL + "Answer:")
        lg = model(mx.array([ids]))[0, -1, :]
        if noise:
            lg = lg + mx.random.normal(lg.shape) * noise
        lp = lg - mx.logsumexp(lg)
        out.append(tuple(float(lp[tokenizer.encode(" " + L)[0]]) for L in labs))
    return out


@pytest.fixture(scope="module")
def quantized_cell():
    items = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    snapshot_download(SRC)
    shutil.rmtree(OUT, ignore_errors=True)
    convert(hf_path=SRC, mlx_path=OUT, quantize=True, q_bits=4, q_group_size=64, q_mode="affine")
    model, tokenizer = load(OUT)
    yield model, tokenizer, items
    shutil.rmtree(OUT, ignore_errors=True)


def test_repeated_inference_is_bit_identical(quantized_cell):
    model, tokenizer, items = quantized_cell
    a = _logprobs(model, tokenizer, items)
    b = _logprobs(model, tokenizer, items)
    assert a == b, "same cell run twice produced different log-probs — PREREG §4.6.6 violated"


def test_injected_noise_is_detected(quantized_cell):
    """Mutation proof (PROTOKOL Kural 4): if 1e-6 noise silently passed as 'identical',
    the comparison above would be vacuous."""
    model, tokenizer, items = quantized_cell
    a = _logprobs(model, tokenizer, items)
    noisy = _logprobs(model, tokenizer, items, noise=1e-6)
    assert a != noisy, "1e-6 noise was not detected — the equality check is vacuous"
