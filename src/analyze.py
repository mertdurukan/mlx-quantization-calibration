"""src/analyze.py — PREREG §5 tables + figures (SPEC §3).

Reads every results/cells/*.parquet + results/meta/*.json and writes exactly
the four pre-registered tables and three figures, plus the PREREG §4.2
floor-control disclosure (its own file, never merged into tables 1-4) and a
verdicts.json with one PASS/FAIL per hypothesis. Nothing here tests a
hypothesis not stated in PREREG §3 (Görev 11 checks this by review, not by
code).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.config as config
import src.metrics as metrics

LADDER_ORDER = [
    "affine_b8_g64", "affine_b6_g64", "affine_b5_g64",
    "affine_b4_g64", "affine_b3_g64", "affine_b2_g64",
]
LADDER_BITS = {"affine_b8_g64": 8, "affine_b6_g64": 6, "affine_b5_g64": 5,
               "affine_b4_g64": 4, "affine_b3_g64": 3, "affine_b2_g64": 2}

MIXED_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]
MIXED_COMPONENTS = {
    "mixed_2_6": ("affine_b2_g64", "affine_b6_g64"),
    "mixed_3_4": ("affine_b3_g64", "affine_b4_g64"),
    "mixed_3_6": ("affine_b3_g64", "affine_b6_g64"),
    "mixed_4_6": ("affine_b4_g64", "affine_b6_g64"),
}

_QUANTIZED_CONDITIONS = [tag for tag, *_ in config.CONDITIONS if tag != "bf16"]


# ---------------------------------------------------------------------------
# Critical verdict logic (SPEC §3, tests/test_analyze.py)
# ---------------------------------------------------------------------------

def _intervals_overlap(lo1, hi1, lo2, hi2) -> bool:
    return lo1 <= hi2 and lo2 <= hi1


def _paired_bootstrap_delta(base_correct, base_conf, other_correct, other_conf,
                             metric_fn, n=config.BOOTSTRAP_N, seed=config.SEED,
                             **metric_kwargs):
    base_correct = np.asarray(base_correct, dtype=float)
    base_conf = np.asarray(base_conf, dtype=float)
    other_correct = np.asarray(other_correct, dtype=float)
    other_conf = np.asarray(other_conf, dtype=float)

    point = (
        metric_fn(other_correct, other_conf, **metric_kwargs)
        - metric_fn(base_correct, base_conf, **metric_kwargs)
    )

    rng = np.random.default_rng(seed)
    n_items = len(base_correct)
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_items, n_items)
        deltas[i] = (
            metric_fn(other_correct[idx], other_conf[idx], **metric_kwargs)
            - metric_fn(base_correct[idx], base_conf[idx], **metric_kwargs)
        )
    lo, hi = np.percentile(deltas, [config.CI_LOW, config.CI_HIGH])
    return float(lo), float(point), float(hi)


def _h1_ladder_verdict(rows: list) -> dict:
    violations = []
    for a, b in zip(rows, rows[1:]):
        if b["point"] < a["point"] and not _intervals_overlap(a["lo"], a["hi"], b["lo"], b["hi"]):
            violations.append((a["bits"], b["bits"]))
    return {"monotone": len(violations) == 0, "violations": violations}


def _h2_direction_verdict(rows: list) -> dict:
    def _key(r):
        return (r["model"], r["benchmark"], r["condition"])

    confirming = [
        _key(r) for r in rows
        if r["delta_intercept_hi"] < 0 and r["delta_conf_incorrect_lo"] > 0
    ]
    contradicting = [
        _key(r) for r in rows
        if r["delta_intercept_lo"] > 0 or r["delta_conf_incorrect_hi"] < 0
    ]
    return {
        "direction_confirmed": len(contradicting) == 0 and len(confirming) > 0,
        "confirming_cells": confirming,
        "contradicting_cells": contradicting,
    }


def _h3_mode_verdict(rows: list) -> dict:
    differing = [(r["model"], r["benchmark"]) for r in rows if r["differs"]]
    return {"mode_matters": len(differing) > 0, "differing_cells": differing}


def _h4_recipe_verdict(recipe_lo, recipe_hi, comp_a_point, comp_b_point) -> bool:
    lo_range, hi_range = sorted([comp_a_point, comp_b_point])
    return _intervals_overlap(recipe_lo, recipe_hi, lo_range, hi_range)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cells(results_dir: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted((results_dir / "cells").glob("*.parquet"))]
    if not frames:
        raise RuntimeError(f"no cells found under {results_dir / 'cells'} — run the grid first")
    cells = pd.concat(frames, ignore_index=True)
    return cells[cells["status"] == "ok"]


def _load_meta(results_dir: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted((results_dir / "meta").glob("*.json"))]
    return pd.DataFrame(rows)


def _eligible_main_models(results_dir: Path) -> list:
    """Models entering tables 1-4: eligible AND not the floor control (PREREG §4.2)."""
    eligibility = json.loads((results_dir / "eligibility.json").read_text())
    return [
        model for model, record in eligibility.items()
        if record.get("eligible") and record.get("role") != "floor_control"
    ]


def _floor_control_models(results_dir: Path) -> list:
    eligibility = json.loads((results_dir / "eligibility.json").read_text())
    return [m for m, r in eligibility.items() if r.get("role") == "floor_control"]


def _sub(cells, model, condition, benchmark):
    return cells[
        (cells["model"] == model) & (cells["condition"] == condition) & (cells["benchmark"] == benchmark)
    ]


def _sorted_arrays(sub: pd.DataFrame):
    sub = sub.sort_values("item_id")
    return sub["is_correct"].astype(float).to_numpy(), sub["conf_pred"].astype(float).to_numpy()


def _metric_ci_block(correct, conf, prefix, metric_fn, **kwargs):
    lo, point, hi = metrics.bootstrap_ci(correct, conf, metric_fn, **kwargs)
    return {f"{prefix}_lo": lo, prefix: point, f"{prefix}_hi": hi}


def _effective_bits(meta: pd.DataFrame, cell_id: str):
    row = meta[meta["cell_id"] == cell_id]
    if row.empty:
        return None
    return row.iloc[0]["effective_bits"]


# ---------------------------------------------------------------------------
# Table 1 (H1) — bit ladder, paired against bf16, monotonicity verdict
# ---------------------------------------------------------------------------

def build_table1(cells: pd.DataFrame, meta: pd.DataFrame, models: list) -> tuple:
    rows = []
    per_cell_verdict = {}
    for model in models:
        for benchmark in config.BENCHMARKS:
            bf16_sub = _sub(cells, model, "bf16", benchmark)
            if bf16_sub.empty:
                continue
            bf16_correct, bf16_conf = _sorted_arrays(bf16_sub)
            bf16_cid = f"{model}__bf16__{benchmark}"
            row = {"model": model, "benchmark": benchmark, "condition": "bf16", "bits": 16,
                   "effective_bits": _effective_bits(meta, bf16_cid)}
            for metric_name, metric_fn in [("ece", metrics.ece), ("slope", metrics.cal_slope),
                                            ("intercept", metrics.cal_intercept), ("brier", metrics.brier)]:
                row.update(_metric_ci_block(bf16_correct, bf16_conf, metric_name, metric_fn))
                row[f"delta_{metric_name}_lo"] = None
                row[f"delta_{metric_name}"] = 0.0
                row[f"delta_{metric_name}_hi"] = None
            rows.append(row)

            ladder_ece_rows = []
            for tag in LADDER_ORDER:
                sub = _sub(cells, model, tag, benchmark)
                if sub.empty:
                    continue
                correct, conf = _sorted_arrays(sub)
                cid = f"{model}__{tag}__{benchmark}"
                row = {"model": model, "benchmark": benchmark, "condition": tag,
                       "bits": LADDER_BITS[tag], "effective_bits": _effective_bits(meta, cid)}
                for metric_name, metric_fn in [("ece", metrics.ece), ("slope", metrics.cal_slope),
                                                ("intercept", metrics.cal_intercept), ("brier", metrics.brier)]:
                    row.update(_metric_ci_block(correct, conf, metric_name, metric_fn))
                    d_lo, d_point, d_hi = _paired_bootstrap_delta(
                        bf16_correct, bf16_conf, correct, conf, metric_fn
                    )
                    row[f"delta_{metric_name}_lo"] = d_lo
                    row[f"delta_{metric_name}"] = d_point
                    row[f"delta_{metric_name}_hi"] = d_hi
                rows.append(row)
                ladder_ece_rows.append({"bits": LADDER_BITS[tag], "point": row["ece"],
                                         "lo": row["ece_lo"], "hi": row["ece_hi"]})

            if ladder_ece_rows:
                per_cell_verdict[f"{model}__{benchmark}"] = _h1_ladder_verdict(ladder_ece_rows)

    df = pd.DataFrame(rows)
    monotone_flags = [v["monotone"] for v in per_cell_verdict.values()]
    overall_pass = bool(monotone_flags) and all(monotone_flags)
    verdict = {
        "hypothesis": "H1",
        "statement": "ECE increases monotonically as bit-width decreases from 8 to 2 bits.",
        "overall_pass": overall_pass,
        "per_model_benchmark": per_cell_verdict,
    }
    return df, verdict


# ---------------------------------------------------------------------------
# Table 2 (H2) — confidence direction, all quantized conditions vs bf16
# ---------------------------------------------------------------------------

def build_table2(cells: pd.DataFrame, models: list) -> tuple:
    rows = []
    verdict_rows = []
    for model in models:
        for benchmark in config.BENCHMARKS:
            bf16_sub = _sub(cells, model, "bf16", benchmark)
            if bf16_sub.empty:
                continue
            bf16_correct, bf16_conf = _sorted_arrays(bf16_sub)

            for condition in ["bf16"] + _QUANTIZED_CONDITIONS:
                sub = _sub(cells, model, condition, benchmark)
                if sub.empty:
                    continue
                correct, conf = _sorted_arrays(sub)
                mean_incorrect, mean_correct = metrics.mean_conf_by_correctness(correct, conf)
                n_above_threshold = int(np.sum(conf > config.OVERCONF_THRESHOLD))
                if n_above_threshold == 0:
                    # overconfidence_rate is undefined (0/0) when no item ever crosses the
                    # threshold — a real finding at extreme compression (e.g. qwen2.5-3b's
                    # 2-bit conditions never exceed 0.90 confidence), not a bootstrap bug.
                    # Reporting a bare NaN here would look like missing data; the count makes
                    # the reason explicit instead.
                    overconf_lo = overconf_point = overconf_hi = None
                else:
                    overconf_lo, overconf_point, overconf_hi = metrics.bootstrap_ci(
                        correct, conf, metrics.overconfidence_rate
                    )
                row = {
                    "model": model, "benchmark": benchmark, "condition": condition,
                    "mean_conf_incorrect": mean_incorrect, "mean_conf_correct": mean_correct,
                    "overconfidence_rate": overconf_point,
                    "overconfidence_rate_lo": overconf_lo, "overconfidence_rate_hi": overconf_hi,
                    "overconfidence_rate_n_qualifying": n_above_threshold,
                }
                if condition == "bf16":
                    row.update({"delta_intercept_lo": None, "delta_intercept": 0.0, "delta_intercept_hi": None,
                                "delta_conf_incorrect_lo": None, "delta_conf_incorrect": 0.0,
                                "delta_conf_incorrect_hi": None})
                else:
                    di_lo, di_point, di_hi = _paired_bootstrap_delta(
                        bf16_correct, bf16_conf, correct, conf, metrics.cal_intercept
                    )
                    dc_lo, dc_point, dc_hi = _paired_bootstrap_delta(
                        bf16_correct, bf16_conf, correct, conf, _mean_conf_incorrect
                    )
                    row.update({"delta_intercept_lo": di_lo, "delta_intercept": di_point,
                                "delta_intercept_hi": di_hi, "delta_conf_incorrect_lo": dc_lo,
                                "delta_conf_incorrect": dc_point, "delta_conf_incorrect_hi": dc_hi})
                    verdict_rows.append({
                        "model": model, "benchmark": benchmark, "condition": condition,
                        "delta_intercept_lo": di_lo, "delta_intercept_hi": di_hi,
                        "delta_conf_incorrect_lo": dc_lo, "delta_conf_incorrect_hi": dc_hi,
                    })
                rows.append(row)

    df = pd.DataFrame(rows)
    h2_verdict = _h2_direction_verdict(verdict_rows)
    verdict = {
        "hypothesis": "H2",
        "statement": "Quantization moves confidence toward incorrect answers relative to bf16.",
        "overall_pass": h2_verdict["direction_confirmed"],
        "confirming_cells": h2_verdict["confirming_cells"],
        "contradicting_cells": h2_verdict["contradicting_cells"],
    }
    return df, verdict


def _mean_conf_incorrect(y_correct, y_conf):
    incorrect, _correct = metrics.mean_conf_by_correctness(y_correct, y_conf)
    return incorrect


# ---------------------------------------------------------------------------
# Table 3 (H3) — affine vs mxfp4 @ 4-bit / group 32
# ---------------------------------------------------------------------------

def build_table3(cells: pd.DataFrame, models: list) -> tuple:
    rows = []
    verdict_rows = []
    for model in models:
        for benchmark in config.BENCHMARKS:
            affine_sub = _sub(cells, model, "affine_b4_g32", benchmark)
            mxfp4_sub = _sub(cells, model, "mxfp4_b4_g32", benchmark)
            if affine_sub.empty or mxfp4_sub.empty:
                continue
            a_correct, a_conf = _sorted_arrays(affine_sub)
            m_correct, m_conf = _sorted_arrays(mxfp4_sub)
            a_lo, a_point, a_hi = metrics.bootstrap_ci(a_correct, a_conf, metrics.ece)
            m_lo, m_point, m_hi = metrics.bootstrap_ci(m_correct, m_conf, metrics.ece)
            differs = not _intervals_overlap(a_lo, a_hi, m_lo, m_hi)
            rows.append({
                "model": model, "benchmark": benchmark,
                "affine_ece_lo": a_lo, "affine_ece": a_point, "affine_ece_hi": a_hi,
                "mxfp4_ece_lo": m_lo, "mxfp4_ece": m_point, "mxfp4_ece_hi": m_hi,
                "differs": differs,
            })
            verdict_rows.append({"model": model, "benchmark": benchmark, "differs": differs})

    df = pd.DataFrame(rows)
    h3_verdict = _h3_mode_verdict(verdict_rows)
    verdict = {
        "hypothesis": "H3",
        "statement": "affine vs mxfp4 differ in ECE at matched 4-bit/group-32.",
        "overall_pass": h3_verdict["mode_matters"],
        "differing_cells": h3_verdict["differing_cells"],
    }
    return df, verdict


# ---------------------------------------------------------------------------
# Table 4 (H4, secondary) — mixed recipes vs component uniform bit-widths
# ---------------------------------------------------------------------------

def build_table4(cells: pd.DataFrame, models: list) -> tuple:
    rows = []
    falsified_recipes = []
    for model in models:
        for benchmark in config.BENCHMARKS:
            for recipe in MIXED_RECIPES:
                recipe_sub = _sub(cells, model, recipe, benchmark)
                comp_a_tag, comp_b_tag = MIXED_COMPONENTS[recipe]
                comp_a_sub = _sub(cells, model, comp_a_tag, benchmark)
                comp_b_sub = _sub(cells, model, comp_b_tag, benchmark)
                if recipe_sub.empty or comp_a_sub.empty or comp_b_sub.empty:
                    continue
                r_correct, r_conf = _sorted_arrays(recipe_sub)
                a_correct, a_conf = _sorted_arrays(comp_a_sub)
                b_correct, b_conf = _sorted_arrays(comp_b_sub)
                r_lo, r_point, r_hi = metrics.bootstrap_ci(r_correct, r_conf, metrics.ece)
                _, a_point, _ = metrics.bootstrap_ci(a_correct, a_conf, metrics.ece)
                _, b_point, _ = metrics.bootstrap_ci(b_correct, b_conf, metrics.ece)
                not_falsified = _h4_recipe_verdict(r_lo, r_hi, a_point, b_point)
                rows.append({
                    "model": model, "benchmark": benchmark, "recipe": recipe,
                    "component_a": comp_a_tag, "component_b": comp_b_tag,
                    "recipe_ece_lo": r_lo, "recipe_ece": r_point, "recipe_ece_hi": r_hi,
                    "component_a_ece": a_point, "component_b_ece": b_point,
                    "within_range": not_falsified,
                })
                if not not_falsified:
                    falsified_recipes.append((model, benchmark, recipe))

    df = pd.DataFrame(rows)
    verdict = {
        "hypothesis": "H4",
        "statement": "Each mixed_a_b recipe's ECE lies between the uniform a-bit and b-bit conditions.",
        "overall_pass": len(falsified_recipes) == 0 and len(rows) > 0,
        "falsified_cells": falsified_recipes,
    }
    return df, verdict


# ---------------------------------------------------------------------------
# Floor control (PREREG §4.2 required disclosure, not one of the four tables)
# ---------------------------------------------------------------------------

def build_floor_control_table(cells: pd.DataFrame, meta: pd.DataFrame, floor_models: list) -> pd.DataFrame:
    rows = []
    for model in floor_models:
        for benchmark in config.BENCHMARKS:
            for condition_tag, *_ in config.CONDITIONS:
                sub = _sub(cells, model, condition_tag, benchmark)
                if sub.empty:
                    continue
                correct, conf = _sorted_arrays(sub)
                cid = f"{model}__{condition_tag}__{benchmark}"
                row = {"model": model, "benchmark": benchmark, "condition": condition_tag,
                       "effective_bits": _effective_bits(meta, cid),
                       "accuracy": float(np.mean(correct))}
                row.update(_metric_ci_block(correct, conf, "ece", metrics.ece))
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figure1(cells: pd.DataFrame, models: list, out_path: Path) -> None:
    """Calibration curves (predicted confidence vs empirical accuracy per
    equal-mass bin), bf16 vs the full g64 bit ladder, one panel per model
    (arc_challenge only — the benchmark figures fix in advance to keep the
    grid readable, PREREG places no per-figure benchmark requirement)."""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), squeeze=False)
    for ax, model in zip(axes[0], models):
        for condition in ["bf16"] + LADDER_ORDER:
            sub = _sub(cells, model, condition, "arc_challenge")
            if sub.empty:
                continue
            correct, conf = _sorted_arrays(sub)
            order = np.argsort(conf, kind="stable")
            correct_bins = np.array_split(correct[order], config.ECE_N_BINS)
            conf_bins = np.array_split(conf[order], config.ECE_N_BINS)
            xs = [float(np.mean(b)) for b in conf_bins if len(b)]
            ys = [float(np.mean(b)) for b in correct_bins if len(b)]
            ax.plot(xs, ys, marker="o", label=condition, alpha=0.8)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted confidence (bin)")
        ax.set_ylabel("empirical accuracy (bin)")
        ax.set_title(f"{model} — arc_challenge")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_figure2(cells: pd.DataFrame, meta: pd.DataFrame, models: list, out_path: Path) -> None:
    """ECE as a function of EFFECTIVE bits/weight (not nominal), bf16 + full
    g64 ladder, one line per model x benchmark."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in models:
        for benchmark in config.BENCHMARKS:
            xs, ys = [], []
            for condition in ["bf16"] + LADDER_ORDER:
                sub = _sub(cells, model, condition, benchmark)
                if sub.empty:
                    continue
                correct, conf = _sorted_arrays(sub)
                cid = f"{model}__{condition}__{benchmark}"
                bits = _effective_bits(meta, cid)
                if bits is None:
                    continue
                xs.append(bits)
                ys.append(metrics.ece(correct, conf))
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            ax.plot(xs, ys, marker="o", label=f"{model} / {benchmark}", alpha=0.8)
    ax.set_xlabel("effective bits/weight")
    ax.set_ylabel("ECE (15 equal-mass bins)")
    ax.set_title("ECE vs effective bit-width")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_figure3(cells: pd.DataFrame, models: list, out_path: Path) -> None:
    """Confidence distribution on correct vs incorrect predictions,
    bf16 vs 4-bit (affine_b4_g64) vs 2-bit (affine_b2_g64), pooled across
    models (arc_challenge only, same rationale as Figure 1)."""
    conditions = ["bf16", "affine_b4_g64", "affine_b2_g64"]
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4.5), squeeze=False)
    for ax, condition in zip(axes[0], conditions):
        correct_confs, incorrect_confs = [], []
        for model in models:
            sub = _sub(cells, model, condition, "arc_challenge")
            if sub.empty:
                continue
            correct, conf = _sorted_arrays(sub)
            correct_confs.append(conf[correct == 1])
            incorrect_confs.append(conf[correct == 0])
        correct_confs = np.concatenate(correct_confs) if correct_confs else np.array([])
        incorrect_confs = np.concatenate(incorrect_confs) if incorrect_confs else np.array([])
        bins = np.linspace(0, 1, 21)
        ax.hist(correct_confs, bins=bins, alpha=0.6, label="correct", density=True)
        ax.hist(incorrect_confs, bins=bins, alpha=0.6, label="incorrect", density=True)
        ax.set_xlim(0, 1)
        ax.set_xlabel("confidence")
        ax.set_title(condition)
    axes[0][0].set_ylabel("density")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Confidence distribution: correct vs incorrect (pooled models, arc_challenge)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(results_dir: str = "results") -> None:
    results_path = Path(results_dir)
    tables_dir = results_path / "tables"
    figures_dir = results_path / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cells = _load_cells(results_path)
    meta = _load_meta(results_path)
    models = _eligible_main_models(results_path)
    floor_models = _floor_control_models(results_path)

    table1, v1 = build_table1(cells, meta, models)
    table2, v2 = build_table2(cells, models)
    table3, v3 = build_table3(cells, models)
    table4, v4 = build_table4(cells, models)
    floor_table = build_floor_control_table(cells, meta, floor_models)

    table1.to_csv(tables_dir / "table1_h1_bit_ladder.csv", index=False)
    table2.to_csv(tables_dir / "table2_h2_confidence_direction.csv", index=False)
    table3.to_csv(tables_dir / "table3_h3_mode_contrast.csv", index=False)
    table4.to_csv(tables_dir / "table4_h4_mixed_recipes.csv", index=False)
    floor_table.to_csv(tables_dir / "table5_floor_control.csv", index=False)

    (tables_dir / "verdicts.json").write_text(json.dumps(
        {"H1": v1, "H2": v2, "H3": v3, "H4": v4}, indent=2, default=str
    ))

    make_figure1(cells, models, figures_dir / "figure1_calibration_curves.png")
    make_figure2(cells, meta, models, figures_dir / "figure2_ece_vs_effective_bits.png")
    make_figure3(cells, models, figures_dir / "figure3_confidence_distribution.png")


if __name__ == "__main__":
    main()
