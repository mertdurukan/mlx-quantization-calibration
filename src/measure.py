"""src/measure.py — model + items -> per-item log-probs (SPEC §3)."""
import time
from pathlib import Path

import mlx.core as mx
import pandas as pd
from mlx_lm import load

from . import config

NL = "\n"


def _format_prompt(template: str, item) -> str:
    options_block = NL.join(f"{label}) {text}" for label, text in zip(item.labels, item.options))
    return template.format(question=item.question, options=options_block)


def _score_item(model, tokenizer, template, item) -> dict:
    prompt = _format_prompt(template, item)
    ids = tokenizer.encode(prompt)

    t0 = time.perf_counter()
    logits = model(mx.array([ids]))[0, -1, :]
    logprobs = logits - mx.logsumexp(logits)
    option_logprobs = [float(logprobs[tokenizer.encode(" " + label)[0]]) for label in item.labels]
    latency_ms = (time.perf_counter() - t0) * 1000

    probs = mx.softmax(mx.array(option_logprobs)).tolist()
    pred_idx = option_logprobs.index(max(option_logprobs))

    return {
        "item_id": item.item_id,
        "n_options": len(item.options),
        "answer_idx": item.answer_idx,
        "pred_idx": pred_idx,
        "is_correct": pred_idx == item.answer_idx,
        "conf_pred": probs[pred_idx],
        "conf_true": probs[item.answer_idx],
        "logprobs": option_logprobs,
        "latency_ms": latency_ms,
        "status": "ok",
        "error": None,
    }


def _failed_row(item, exc: Exception) -> dict:
    return {
        "item_id": item.item_id,
        "n_options": len(item.options),
        "answer_idx": item.answer_idx,
        "pred_idx": None,
        "is_correct": None,
        "conf_pred": None,
        "conf_true": None,
        "logprobs": None,
        "latency_ms": None,
        "status": "failed",
        "error": str(exc),
    }


def run_cell(model_path: str, items: list) -> pd.DataFrame:
    """Score items with ONE model. Returns one row PER ITEM (schema in SPEC §4)."""
    model, tokenizer = load(model_path)
    template = Path(config.PROMPT_FILE).read_text()

    for item in items[: config.N_WARMUP]:
        try:
            _score_item(model, tokenizer, template, item)
        except Exception:
            pass  # warmup exists only to prime lazy weights/kernels; never scored either way

    rows = []
    for item in items:
        try:
            rows.append(_score_item(model, tokenizer, template, item))
        except Exception as exc:
            rows.append(_failed_row(item, exc))
    return pd.DataFrame(rows)
