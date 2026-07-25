"""src/quantize.py — condition -> quantized model on disk (SPEC §3)."""
import io
import re
import shutil
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import mlx_lm
from huggingface_hub import snapshot_download

from . import config

_CONDITIONS = {tag: (mode, bits, group_size) for tag, mode, bits, group_size in config.CONDITIONS}
_BPW_RE = re.compile(r"Quantized model with ([\d.]+) bits per weight")


class _Tee(io.StringIO):
    """Captures conversion output for parsing while still printing it."""

    def __init__(self, echo):
        super().__init__()
        self._echo = echo

    def write(self, s):
        self._echo.write(s)
        return super().write(s)


def _resolve_source(model_key: str) -> str:
    src = config.MAIN_MODELS.get(model_key) or config.FLOOR_CONTROL.get(model_key)
    if src is None:
        raise KeyError(f"unknown model_key {model_key!r}")
    return src


def _dir_size_mb(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / 1e6


def build(model_key: str, condition_tag: str, out_dir: str) -> dict:
    """Produce one quantized model on disk from the single bf16 source (SPEC §3)."""
    if condition_tag not in _CONDITIONS:
        raise KeyError(f"unknown condition_tag {condition_tag!r}")
    mode, bits, group_size = _CONDITIONS[condition_tag]

    src = _resolve_source(model_key)
    snapshot_path = snapshot_download(src)

    if mode is None:  # bf16 reference: no quantization
        return {
            "path": snapshot_path,
            "effective_bits": 16.0,
            "size_mb": _dir_size_mb(snapshot_path),
            "convert_seconds": 0.0,
        }

    kwargs = dict(hf_path=src, mlx_path=out_dir, quantize=True)
    if mode == "recipe":
        kwargs.update(q_mode="affine", q_group_size=None, q_bits=None, quant_predicate=condition_tag)
    else:
        kwargs.update(q_mode=mode, q_group_size=group_size, q_bits=bits)

    buf = _Tee(sys.stdout)
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        mlx_lm.convert(**kwargs)
    convert_seconds = time.perf_counter() - t0

    log = buf.getvalue()
    match = _BPW_RE.search(log)
    if match is None:
        raise RuntimeError(f"effective bits not found in conversion log:\n{log}")

    return {
        "path": out_dir,
        "effective_bits": float(match.group(1)),
        "size_mb": _dir_size_mb(out_dir),
        "convert_seconds": convert_seconds,
    }


def teardown(path: str) -> None:
    """Delete a converted model directory. Never touches the HF cache."""
    shutil.rmtree(path)
