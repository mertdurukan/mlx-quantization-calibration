"""Load + sample + normalize ARC-Challenge / MMLU (SPEC §3)."""
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

import src.config as config


@dataclass(frozen=True)
class Item:
    item_id: str
    question: str
    options: list[str]        # option TEXTS, in order
    labels: list[str]         # option LETTERS, e.g. ["A","B","C","D"]
    answer_idx: int           # index into options/labels


def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def _load_arc_challenge() -> list[Item]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = sorted(ds, key=lambda r: r["id"])[: config.N_ITEMS]
    items = []
    for row in rows:
        options = row["choices"]["text"]
        answer_idx = row["choices"]["label"].index(row["answerKey"])
        items.append(
            Item(
                item_id=row["id"],
                question=row["question"],
                options=options,
                labels=_letters(len(options)),
                answer_idx=answer_idx,
            )
        )
    return items


def _load_mmlu() -> list[Item]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    subjects = ds["subject"]

    indices_by_subject: dict[str, list[int]] = {}
    for i, subject in enumerate(subjects):
        indices_by_subject.setdefault(subject, []).append(i)

    subject_names = sorted(indices_by_subject)
    total = len(ds)

    quotas = {s: (config.N_ITEMS * len(indices_by_subject[s])) // total for s in subject_names}
    remainder = config.N_ITEMS - sum(quotas.values())
    for s in subject_names[:remainder]:
        quotas[s] += 1

    rng = np.random.default_rng(config.SEED)
    items = []
    for s in subject_names:
        pool = indices_by_subject[s]
        chosen = rng.choice(pool, size=quotas[s], replace=False)
        for i in sorted(int(x) for x in chosen):
            row = ds[i]
            options = row["choices"]
            items.append(
                Item(
                    item_id=f"mmlu_{s}_{i}",
                    question=row["question"],
                    options=options,
                    labels=_letters(len(options)),
                    answer_idx=int(row["answer"]),
                )
            )
    return items


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
    if benchmark == "arc_challenge":
        return _load_arc_challenge()
    if benchmark == "mmlu":
        return _load_mmlu()
    raise ValueError(f"unknown benchmark: {benchmark!r}")
