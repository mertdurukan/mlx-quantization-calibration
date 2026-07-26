"""Known-answer tests for the hypothesis-verdict logic in src/analyze.py
(SPEC §3, YAPILACAKLAR Görev 10).

Written BEFORE src/analyze.py's bodies exist (PROTOKOL Kural 3): test yaz ->
review -> onay -> implementasyon. Do not weaken any assertion here to make
an implementation pass.

Scope: only the five pure, argument-in/value-out functions that decide
whether H1-H4 PASS or FAIL (SPEC §3 `src/analyze.py`). These are the
functions whose bugs would silently corrupt a published scientific verdict
(PROTOKOL Kural 4: a critical protection test must be proven able to fail).
The table/figure-writing side of analyze.py is data-shaped, not logic-shaped,
and is verified functionally against the real 114-cell grid instead
(PROTOKOL Kural 6), not with known-answer tests here.
"""
import numpy as np
import pytest

import src.analyze as analyze
import src.config as config


# ---------------------------------------------------------------------------
# _intervals_overlap
# ---------------------------------------------------------------------------

def test_intervals_overlap_true_when_overlapping():
    assert analyze._intervals_overlap(0.0, 0.5, 0.3, 0.8) is True


def test_intervals_overlap_true_when_touching_at_boundary():
    # Inclusive: sharing exactly one point counts as overlap.
    assert analyze._intervals_overlap(0.0, 0.5, 0.5, 1.0) is True


def test_intervals_overlap_false_when_disjoint_a_below_b():
    assert analyze._intervals_overlap(0.0, 0.2, 0.3, 0.5) is False


def test_intervals_overlap_false_when_disjoint_b_below_a():
    assert analyze._intervals_overlap(0.3, 0.5, 0.0, 0.2) is False


def test_intervals_overlap_true_when_identical():
    assert analyze._intervals_overlap(0.1, 0.4, 0.1, 0.4) is True


def test_intervals_overlap_true_when_one_contains_other():
    assert analyze._intervals_overlap(0.0, 1.0, 0.4, 0.6) is True


# ---------------------------------------------------------------------------
# _paired_bootstrap_delta
# ---------------------------------------------------------------------------

def _mean_metric(y_correct, y_conf):
    # A trivial metric_fn: ignores correctness, returns mean confidence.
    # Lets the point estimate be checked against a hand-computable number.
    return float(np.mean(y_conf))


def test_paired_bootstrap_delta_point_is_full_sample_difference_not_bootstrap_mean():
    # The delta must vary across resamples, otherwise the bootstrap mean equals
    # the full-sample difference by construction and an implementation that
    # reports the bootstrap mean passes anyway. An independent mutation run on
    # 2026-07-26 showed exactly that: the earlier version of this test used a
    # constant +0.1 shift and the bootstrap-mean mutant survived it.
    rng = np.random.default_rng(1)
    n = 200
    base_conf = rng.uniform(0, 1, n)
    other_conf = np.clip(base_conf + rng.normal(0.1, 0.25, n), 0, 1)
    base_correct = np.ones(n)
    other_correct = np.ones(n)

    lo, point, hi = analyze._paired_bootstrap_delta(
        base_correct, base_conf, other_correct, other_conf, _mean_metric,
        n=500, seed=config.SEED,
    )
    expected_point = float(np.mean(other_conf)) - float(np.mean(base_conf))
    assert point == pytest.approx(expected_point, abs=1e-9)
    assert lo <= point <= hi

    # Pin the distinction directly: recompute the bootstrap mean the same way
    # the function does and require the reported point to differ from it.
    boot_rng = np.random.default_rng(config.SEED)
    deltas = []
    for _ in range(500):
        idx = boot_rng.integers(0, n, n)
        deltas.append(_mean_metric(other_correct[idx], other_conf[idx])
                      - _mean_metric(base_correct[idx], base_conf[idx]))
    bootstrap_mean = float(np.mean(deltas))
    assert abs(point - bootstrap_mean) > 1e-6, (
        "point estimate is indistinguishable from the bootstrap mean on this data"
    )


def test_paired_bootstrap_delta_is_exactly_zero_when_arms_are_identical():
    # Paired resampling draws the SAME indices for both arms every time, so
    # if other == base row-for-row, every resampled delta is exactly 0 -
    # not just "close to 0 in expectation". A shared-index bug (e.g.
    # resampling base and other independently) would violate this.
    rng = np.random.default_rng(2)
    conf = rng.uniform(0, 1, 150)
    correct = (rng.uniform(0, 1, 150) > 0.5).astype(float)

    lo, point, hi = analyze._paired_bootstrap_delta(
        correct, conf, correct, conf, _mean_metric, n=300, seed=config.SEED,
    )
    assert point == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_paired_bootstrap_delta_forwards_kwargs_to_metric_fn():
    def _thresholded_mean(y_correct, y_conf, offset):
        return float(np.mean(y_conf)) + offset

    base_conf = np.array([0.2, 0.4, 0.6, 0.8])
    other_conf = np.array([0.3, 0.5, 0.7, 0.9])
    base_correct = other_correct = np.array([1.0, 0.0, 1.0, 0.0])

    lo, point, hi = analyze._paired_bootstrap_delta(
        base_correct, base_conf, other_correct, other_conf,
        _thresholded_mean, n=50, seed=config.SEED, offset=10.0,
    )
    # The +10.0 offset cancels in the difference; forwarding is only
    # observable via the fact that calling WITHOUT it would TypeError.
    assert point == pytest.approx(0.1, abs=1e-9)


def test_paired_bootstrap_delta_missing_required_kwarg_raises():
    def _needs_kwarg(y_correct, y_conf, required):
        return float(np.mean(y_conf)) + required

    conf = np.array([0.2, 0.4, 0.6, 0.8])
    correct = np.array([1.0, 0.0, 1.0, 0.0])
    with pytest.raises(TypeError):
        analyze._paired_bootstrap_delta(correct, conf, correct, conf, _needs_kwarg, n=10, seed=0)


# ---------------------------------------------------------------------------
# _h1_ladder_verdict
# ---------------------------------------------------------------------------

def test_h1_monotone_increasing_ece_passes():
    rows = [
        {"bits": 8, "point": 0.05, "lo": 0.03, "hi": 0.07},
        {"bits": 6, "point": 0.07, "lo": 0.05, "hi": 0.09},
        {"bits": 5, "point": 0.09, "lo": 0.07, "hi": 0.11},
        {"bits": 4, "point": 0.12, "lo": 0.10, "hi": 0.14},
        {"bits": 3, "point": 0.20, "lo": 0.17, "hi": 0.23},
        {"bits": 2, "point": 0.35, "lo": 0.30, "hi": 0.40},
    ]
    verdict = analyze._h1_ladder_verdict(rows)
    assert verdict == {"monotone": True, "violations": []}


def test_h1_real_reversal_with_non_overlapping_cis_fails():
    rows = [
        {"bits": 8, "point": 0.05, "lo": 0.03, "hi": 0.07},
        {"bits": 6, "point": 0.20, "lo": 0.18, "hi": 0.22},
        {"bits": 5, "point": 0.05, "lo": 0.03, "hi": 0.07},  # real dip: CIs don't overlap with bits=6
        {"bits": 4, "point": 0.30, "lo": 0.27, "hi": 0.33},
    ]
    verdict = analyze._h1_ladder_verdict(rows)
    assert verdict["monotone"] is False
    assert verdict["violations"] == [(6, 5)]


def test_h1_reversal_within_overlapping_cis_is_noise_and_passes():
    rows = [
        {"bits": 8, "point": 0.10, "lo": 0.05, "hi": 0.15},
        {"bits": 6, "point": 0.11, "lo": 0.06, "hi": 0.16},  # tiny dip vs 8, but CIs overlap heavily
        {"bits": 5, "point": 0.09, "lo": 0.04, "hi": 0.14},  # dip vs 6, but CIs overlap
        {"bits": 4, "point": 0.20, "lo": 0.17, "hi": 0.23},
    ]
    verdict = analyze._h1_ladder_verdict(rows)
    assert verdict == {"monotone": True, "violations": []}


def test_h1_multiple_real_reversals_are_all_reported():
    rows = [
        {"bits": 8, "point": 0.30, "lo": 0.27, "hi": 0.33},
        {"bits": 6, "point": 0.05, "lo": 0.03, "hi": 0.07},  # dip vs 8
        {"bits": 5, "point": 0.40, "lo": 0.37, "hi": 0.43},
        {"bits": 4, "point": 0.10, "lo": 0.08, "hi": 0.12},  # dip vs 5
    ]
    verdict = analyze._h1_ladder_verdict(rows)
    assert verdict["monotone"] is False
    # Expectation widened on 2026-07-26 when the verdict was corrected to the
    # pre-registered criterion (ordering over ALL pairs, PREREG H1) instead of
    # adjacent steps only. (8, 4) is a genuine CI-confirmed reversal — 0.30
    # [0.27,0.33] vs 0.10 [0.08,0.12] — that the adjacent-only rule never looked
    # at. This is a stricter expectation, not a relaxed one.
    assert verdict["violations"] == [(8, 6), (8, 4), (5, 4)]


# ---------------------------------------------------------------------------
# _h2_direction_verdict
# ---------------------------------------------------------------------------

def _h2_cell(model, benchmark, condition, di_lo, di_hi, dc_lo, dc_hi):
    return {
        "model": model, "benchmark": benchmark, "condition": condition,
        "delta_intercept_lo": di_lo, "delta_intercept_hi": di_hi,
        "delta_conf_incorrect_lo": dc_lo, "delta_conf_incorrect_hi": dc_hi,
    }


def test_h2_all_cells_confirm_direction():
    rows = [
        _h2_cell("qwen2.5-1.5b", "arc_challenge", "affine_b4_g64", -0.5, -0.1, 0.02, 0.10),
        _h2_cell("qwen2.5-1.5b", "arc_challenge", "affine_b2_g64", -0.8, -0.3, 0.05, 0.20),
    ]
    verdict = analyze._h2_direction_verdict(rows)
    assert verdict["direction_confirmed"] is True
    assert verdict["contradicting_cells"] == []
    assert len(verdict["confirming_cells"]) == 2


def test_h2_one_contradicting_cell_falsifies_overall_even_if_others_confirm():
    rows = [
        _h2_cell("qwen2.5-1.5b", "arc_challenge", "affine_b4_g64", -0.5, -0.1, 0.02, 0.10),
        # intercept moved significantly POSITIVE -> wrong direction.
        _h2_cell("qwen2.5-1.5b", "mmlu", "affine_b4_g64", 0.05, 0.30, 0.02, 0.10),
    ]
    verdict = analyze._h2_direction_verdict(rows)
    assert verdict["direction_confirmed"] is False
    assert verdict["contradicting_cells"] == [("qwen2.5-1.5b", "mmlu", "affine_b4_g64")]


def test_h2_contradiction_via_confidence_component_alone():
    rows = [
        # intercept direction fine, but confidence-on-wrong significantly FELL.
        _h2_cell("qwen2.5-3b", "arc_challenge", "affine_b2_g64", -0.4, -0.1, -0.15, -0.02),
    ]
    verdict = analyze._h2_direction_verdict(rows)
    assert verdict["direction_confirmed"] is False
    assert verdict["contradicting_cells"] == [("qwen2.5-3b", "arc_challenge", "affine_b2_g64")]


def test_h2_all_inconclusive_cis_spanning_zero_does_not_confirm():
    rows = [
        _h2_cell("qwen2.5-1.5b", "arc_challenge", "affine_b8_g64", -0.2, 0.1, -0.05, 0.05),
    ]
    verdict = analyze._h2_direction_verdict(rows)
    assert verdict["direction_confirmed"] is False
    assert verdict["confirming_cells"] == []
    assert verdict["contradicting_cells"] == []


def test_h2_empty_rows_does_not_falsely_confirm():
    verdict = analyze._h2_direction_verdict([])
    assert verdict == {"direction_confirmed": False, "confirming_cells": [], "contradicting_cells": []}


# ---------------------------------------------------------------------------
# _h3_mode_verdict
# ---------------------------------------------------------------------------

def test_h3_falsified_when_every_cell_overlaps():
    rows = [
        {"model": "qwen2.5-1.5b", "benchmark": "arc_challenge", "differs": False},
        {"model": "qwen2.5-1.5b", "benchmark": "mmlu", "differs": False},
        {"model": "qwen2.5-3b", "benchmark": "arc_challenge", "differs": False},
    ]
    verdict = analyze._h3_mode_verdict(rows)
    assert verdict == {"mode_matters": False, "differing_cells": []}


def test_h3_holds_if_any_single_cell_differs():
    rows = [
        {"model": "qwen2.5-1.5b", "benchmark": "arc_challenge", "differs": False},
        {"model": "qwen2.5-1.5b", "benchmark": "mmlu", "differs": True},
        {"model": "qwen2.5-3b", "benchmark": "arc_challenge", "differs": False},
    ]
    verdict = analyze._h3_mode_verdict(rows)
    assert verdict["mode_matters"] is True
    assert verdict["differing_cells"] == [("qwen2.5-1.5b", "mmlu")]


def test_h3_empty_rows_does_not_falsely_pass():
    # No cells evaluated should never silently read as "mode matters"
    # confirmed for zero evidence.
    verdict = analyze._h3_mode_verdict([])
    assert verdict == {"mode_matters": False, "differing_cells": []}


# ---------------------------------------------------------------------------
# _h4_recipe_verdict
# ---------------------------------------------------------------------------

def test_h4_recipe_ci_inside_component_range_not_falsified():
    # mixed_3_4 between affine_b3 (0.10) and affine_b4 (0.20)
    assert analyze._h4_recipe_verdict(0.13, 0.17, comp_a_point=0.10, comp_b_point=0.20) is True


def test_h4_recipe_ci_entirely_below_range_falsified():
    assert analyze._h4_recipe_verdict(0.01, 0.05, comp_a_point=0.10, comp_b_point=0.20) is False


def test_h4_recipe_ci_entirely_above_range_falsified():
    assert analyze._h4_recipe_verdict(0.25, 0.30, comp_a_point=0.10, comp_b_point=0.20) is False


def test_h4_recipe_ci_partially_overlapping_boundary_not_falsified():
    # CI straddles the lower edge of the range -> still touches the range.
    assert analyze._h4_recipe_verdict(0.08, 0.12, comp_a_point=0.10, comp_b_point=0.20) is True


def test_h4_recipe_range_is_order_independent_in_component_args():
    # comp_a/comp_b may be passed in either order (a isn't necessarily <= b).
    assert analyze._h4_recipe_verdict(0.13, 0.17, comp_a_point=0.20, comp_b_point=0.10) is True
    assert analyze._h4_recipe_verdict(0.01, 0.05, comp_a_point=0.20, comp_b_point=0.10) is False


# --- H1: the pre-registered criterion is an ORDERING, not adjacent steps -----
# PREREG H1: "the ECE ordering across {8, 6, 5, 4, 3, 2} is not monotone
# increasing". An ordering over a set constrains every pair. An adjacent-only
# implementation is strictly weaker and biased toward PASS; an independent audit
# on 2026-07-26 found two real CI-confirmed reversals hidden by it.


def _ladder(*triples):
    """(bits, point, half_width) -> rows in descending bit order."""
    return [{"bits": b, "point": p, "lo": p - w, "hi": p + w} for b, p, w in triples]


def test_h1_flags_a_non_adjacent_reversal_that_adjacent_steps_miss():
    # 8 -> 3 is a CI-confirmed reversal, but every adjacent step overlaps its
    # neighbour, so an adjacent-only rule reports this ladder as monotone.
    rows = _ladder((8, 0.30, 0.02), (6, 0.27, 0.02), (5, 0.24, 0.02),
                   (4, 0.21, 0.02), (3, 0.18, 0.02), (2, 0.40, 0.02))
    # Fixture requirement: the adjacent-only rule must find NOTHING here, so the
    # test can only pass if the all-pairs criterion is what is implemented.
    adjacent_violations = [
        (a["bits"], b["bits"]) for a, b in zip(rows, rows[1:])
        if b["point"] < a["point"]
        and not analyze._intervals_overlap(a["lo"], a["hi"], b["lo"], b["hi"])
    ]
    assert adjacent_violations == [], "fixture must be invisible to an adjacent-only rule"

    verdict = analyze._h1_ladder_verdict(rows)
    assert verdict["monotone"] is False
    assert (8, 3) in verdict["violations"]


def test_h1_still_passes_a_genuinely_monotone_ladder():
    rows = _ladder((8, 0.10, 0.01), (6, 0.12, 0.01), (5, 0.15, 0.01),
                   (4, 0.20, 0.01), (3, 0.30, 0.01), (2, 0.45, 0.01))
    verdict = analyze._h1_ladder_verdict(rows)
    assert verdict["monotone"] is True
    assert verdict["violations"] == []


def test_h1_does_not_flag_a_reversal_whose_intervals_overlap():
    # A dip that the intervals cannot separate is noise, per PREREG's
    # "with 95% intervals excluding the reversal being noise".
    rows = _ladder((8, 0.30, 0.05), (6, 0.28, 0.05), (5, 0.32, 0.05))
    verdict = analyze._h1_ladder_verdict(rows)
    assert verdict["monotone"] is True


def test_h1_reports_no_verdict_when_there_is_nothing_to_order():
    # One surviving ladder condition is absence of evidence. Returning True here
    # would let a degenerate cell contribute a silent PASS to the aggregate.
    assert analyze._h1_ladder_verdict([])["monotone"] is None
    assert analyze._h1_ladder_verdict(_ladder((8, 0.1, 0.01)))["monotone"] is None


# --- undefined intervals must not be answered as booleans -------------------


def test_intervals_overlap_raises_on_non_finite_bounds():
    # Every `<=` against NaN is False, so a NaN bound would read as "these
    # intervals do not overlap" -- which H1/H3/H4 treat as positive evidence,
    # turning an undefined metric into a verdict.
    nan = float("nan")
    for bounds in [(nan, nan, 0.1, 0.2), (0.1, 0.2, nan, nan),
                   (0.1, nan, 0.1, 0.2), (float("inf"), 0.2, 0.1, 0.2)]:
        with pytest.raises(ValueError):
            analyze._intervals_overlap(*bounds)


def test_h1_propagates_undefined_intervals_instead_of_passing():
    nan = float("nan")
    rows = [{"bits": 8, "point": 0.3, "lo": 0.28, "hi": 0.32},
            {"bits": 6, "point": nan, "lo": nan, "hi": nan}]
    with pytest.raises(ValueError):
        analyze._h1_ladder_verdict(rows)
