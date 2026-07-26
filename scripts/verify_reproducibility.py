"""Prove that every reported table regenerates bit-identically from committed data.

`make reproduce` re-runs the whole measurement grid (~6.6 hours of model conversion
and inference). This does the cheap half: it re-derives results/tables/ from the
committed per-item measurements in results/cells/ and checks the output is
byte-identical to what is committed.

That check is what makes two claims falsifiable rather than assertions:

  1. "nothing under results/ is hand-edited" (PROTOKOL Kural 11) -- a hand-edited
     table shows up here as a diff.
  2. "the analysis is deterministic" -- the bootstrap is seeded, so a re-run that
     produced different intervals would mean the seed is not actually controlling
     the resampling.

Usage:
    python scripts/verify_reproducibility.py           # exits non-zero on any diff
    python scripts/verify_reproducibility.py --keep    # leave regenerated files in place

Exit code 0 means every committed table was reproduced exactly.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TABLES = REPO / "results" / "tables"
FIGURES = REPO / "results" / "figures"


def snapshot(dest: Path) -> list[Path]:
    """Copy the committed tables aside before they are overwritten."""
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in TABLES.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"no committed tables found in {TABLES}")
    for p in files:
        shutil.copy2(p, dest / p.name)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true",
                    help="do not restore the committed tables afterwards")
    args = ap.parse_args()

    n_cells = len(list((REPO / "results" / "cells").glob("*.parquet")))
    print(f"committed per-item measurements: {n_cells} cells")
    if n_cells == 0:
        raise SystemExit(
            "results/cells/ is empty -- this check needs the committed measurements"
        )

    tmp = Path(tempfile.mkdtemp(prefix="repro_check_"))
    committed = tmp / "committed"
    original = snapshot(committed)
    print(f"snapshotted {len(original)} committed tables")

    print("re-running python -m src.analyze (bootstrap over items; takes ~10 min)")
    r = subprocess.run(
        [sys.executable, "-m", "src.analyze"],
        cwd=REPO, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout[-4000:], file=sys.stderr)
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit("src.analyze failed; cannot verify reproducibility")

    # numpy warnings here would mean a metric went degenerate on some cell -- worth
    # surfacing rather than swallowing.
    warnings = [ln for ln in (r.stderr or "").split("\n") if "Warning" in ln]
    if warnings:
        print(f"\nWARNINGS from src.analyze ({len(warnings)}):")
        for ln in warnings[:10]:
            print("  " + ln)

    identical, differing, missing = [], [], []
    for p in original:
        regenerated = TABLES / p.name
        if not regenerated.exists():
            missing.append(p.name)
        elif filecmp.cmp(committed / p.name, regenerated, shallow=False):
            identical.append(p.name)
        else:
            differing.append(p.name)

    print(f"\nidentical: {len(identical)}/{len(original)}")
    for name in identical:
        print(f"  ok    {name}")
    for name in differing:
        print(f"  DIFF  {name}")
    for name in missing:
        print(f"  GONE  {name}")

    if not args.keep:
        for p in original:
            shutil.copy2(committed / p.name, TABLES / p.name)
        print("\ncommitted tables restored")

    shutil.rmtree(tmp, ignore_errors=True)

    if differing or missing:
        print("\nFAIL: results/tables/ does not regenerate from results/cells/")
        return 1
    print("\nPASS: every committed table regenerates exactly from committed cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
