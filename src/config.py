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
# SHA-256 of PROMPT_FILE at freeze time (see tests/test_prompt_frozen.py).
# Not listed in SPEC §2's original block; added in the same commit that froze the template
# (SPEC §0 prohibition 3 requires the template committed before any run — this constant is
# how that freeze is mechanically enforced). See GECMIS.md.
PROMPT_SHA256: Final[str] = "4c5420822dcf4f0a3d14b58f6279b727d032ac52fc408e6f444e2bdb8a95c915"

# --- metrics (PREREG §4.5) ---
ECE_N_BINS: Final[int] = 15      # equal-MASS
BOOTSTRAP_N: Final[int] = 2_000
CI_LOW: Final[float] = 2.5
CI_HIGH: Final[float] = 97.5
OVERCONF_THRESHOLD: Final[float] = 0.90   # "high-confidence error" cutoff
