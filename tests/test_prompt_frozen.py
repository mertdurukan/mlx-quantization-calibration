"""PREREG §4.4 / SPEC §0 prohibition 3 — the prompt template is frozen.

If this test fails, the template file was edited after freeze. Do not update
config.PROMPT_SHA256 to make it pass — that defeats the point of the test. A change to the
template is a new experiment; it belongs in DEVIATIONS.md, decided BEFORE any affected run.
"""
import hashlib

import src.config as config


def test_prompt_template_matches_frozen_hash():
    with open(config.PROMPT_FILE, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == config.PROMPT_SHA256, (
        f"prompts/mc_letter.txt has changed since freeze (got {digest}, "
        f"expected {config.PROMPT_SHA256}). See SPEC §0 prohibition 3."
    )
