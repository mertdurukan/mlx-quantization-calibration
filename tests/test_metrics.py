"""Known-answer tests for src/metrics.py (SPEC §3, YAPILACAKLAR Görev 2).

Written BEFORE src/metrics.py exists (SPEC §7 build order, PROTOKOL Kural 3):
test yaz -> bağımsız inceleme -> onay -> implementasyon. Do not weaken any
assertion here to make an implementation pass (Görev 3 acceptance criterion
is `git diff HEAD -- tests/` empty).

Where a "known answer" requires a nontrivial numeric reference (ECE, cal_slope,
cal_intercept), the reference is computed independently in this file — via
the exact equal-mass/`np.array_split` algorithm SPEC §3 mandates for `ece`, or
via a from-scratch Newton-Raphson logistic fit for `cal_slope`/`cal_intercept`
— rather than hand-typing a number, and every synthetic dataset was verified
empirically (scratch script, not committed) before the tolerances below were
fixed. See GECMIS.md "Görev 2" entry for the reasoning behind each construction.
"""
import numpy as np
import pytest

import src.config as config
import src.metrics as metrics


def _reference_equal_mass_ece(y_correct, y_conf, n_bins):
    """Independent re-implementation of SPEC §3's ece algorithm: sort by
    confidence, split into n_bins via np.array_split (guarantees bin sizes
    differ by at most 1), weight each bin's |accuracy - mean confidence| by
    its share of N. Used as ground truth, not imported from src.metrics.
    """
    order = np.argsort(y_conf, kind="stable")
    sorted_correct = np.asarray(y_correct, dtype=float)[order]
    sorted_conf = np.asarray(y_conf, dtype=float)[order]
    correct_bins = np.array_split(sorted_correct, n_bins)
    conf_bins = np.array_split(sorted_conf, n_bins)
    n_total = len(y_conf)
    ece = 0.0
    sizes = []
    for correct_bin, conf_bin in zip(correct_bins, conf_bins):
        sizes.append(len(correct_bin))
        if len(correct_bin) == 0:
            continue
        ece += (len(correct_bin) / n_total) * abs(
            np.mean(correct_bin) - np.mean(conf_bin)
        )
    return ece, sizes


def _logit(p):
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _fit_reference_cal_slope_intercept(y, conf, n_iter=100):
    """From-scratch Newton-Raphson MLE for y ~ sigmoid(b0 + b1*logit(conf)).
    Ground truth for the unconstrained (slope, intercept) logistic fit —
    independent of statsmodels, since statsmodels is not installed (see
    open finding recorded alongside this task).
    """
    x = _logit(np.clip(conf, 1e-6, 1 - 1e-6))
    b0, b1 = 0.0, 1.0
    for _ in range(n_iter):
        p = _sigmoid(b0 + b1 * x)
        w = p * (1 - p)
        g = np.array([np.sum(y - p), np.sum((y - p) * x)])
        h = -np.array(
            [[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]]
        )
        b0, b1 = np.array([b0, b1]) - np.linalg.solve(h, g)
    return b0, b1


def _fit_reference_cal_intercept_offset(y, conf, n_iter=100):
    """Ground truth for the offset model: y ~ sigmoid(b0 + 1*logit(conf)),
    slope frozen at 1 (this is what `cal_intercept` is defined to report,
    per SPEC §3: "statsmodels GLM with offset=").
    """
    x = _logit(np.clip(conf, 1e-6, 1 - 1e-6))
    b0 = 0.0
    for _ in range(n_iter):
        p = _sigmoid(b0 + x)
        w = p * (1 - p)
        b0 -= np.sum(y - p) / -np.sum(w)
    return b0


# --- cal_slope ---------------------------------------------------------


def test_cal_slope_perfect_calibration_is_near_one():
    rng = np.random.default_rng(42)
    n = 20_000
    conf = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, conf).astype(float)
    _, expected_slope = _fit_reference_cal_slope_intercept(y, conf)
    assert abs(expected_slope - 1.0) < 0.05  # sanity-check own reference fit

    slope = metrics.cal_slope(y, conf)
    assert abs(slope - 1.0) < 0.05


def test_cal_slope_overextreme_predictions_is_notably_below_one():
    # y is generated from p_true, but reported confidence overstates it by
    # doubling the log-odds (logit(conf) = 2*logit(p_true)) -> the true
    # slope of y on logit(conf) is ~0.5, clearly and stably < 1.
    rng = np.random.default_rng(7)
    n = 20_000
    p_true = rng.uniform(0.1, 0.9, n)
    y = rng.binomial(1, p_true).astype(float)
    conf = _sigmoid(2 * _logit(p_true))

    slope = metrics.cal_slope(y, conf)
    assert slope < 0.75


# --- cal_intercept -------------------------------------------------------


def test_cal_intercept_perfect_calibration_is_near_zero():
    rng = np.random.default_rng(42)
    n = 20_000
    conf = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, conf).astype(float)

    intercept = metrics.cal_intercept(y, conf)
    assert abs(intercept) < 0.05


def test_cal_intercept_systematic_overconfidence_is_notably_below_zero():
    # Reported confidence is the true log-odds shifted up by +1 (overconfident)
    # while the slope stays 1 -> the offset-model intercept recovers ~ -1.
    rng = np.random.default_rng(11)
    n = 20_000
    p_true = rng.uniform(0.1, 0.9, n)
    y = rng.binomial(1, p_true).astype(float)
    conf = _sigmoid(_logit(p_true) + 1.0)

    intercept = metrics.cal_intercept(y, conf)
    assert intercept < -0.3


# --- ece -------------------------------------------------------------


def test_ece_equal_mass_bins_match_reference_algorithm():
    # N=97 is not divisible by config.ECE_N_BINS=15, forcing bins of two
    # different sizes -> also verifies the "differ by at most 1" guarantee
    # that np.array_split (SPEC's mandated algorithm) provides.
    rng = np.random.default_rng(3)
    n = 97
    conf = rng.uniform(0.01, 0.99, n)
    y = rng.binomial(1, conf).astype(float)

    expected, sizes = _reference_equal_mass_ece(y, conf, config.ECE_N_BINS)
    assert max(sizes) - min(sizes) <= 1

    assert metrics.ece(y, conf) == pytest.approx(expected, abs=1e-9)


def test_ece_perfect_calibration_is_near_zero():
    rng = np.random.default_rng(6)
    n = 50_000
    conf = rng.uniform(0.02, 0.98, n)
    y = rng.binomial(1, conf).astype(float)

    assert metrics.ece(y, conf) < 0.01


def test_ece_balanced_labels_constant_confidence_is_about_point_four():
    # Constant confidence 0.9, exactly balanced 50/50 labels. Labels are
    # seed-shuffled, NOT left sorted -- the sibling study's ECE test broke
    # even correct code because it generated y in sorted order (PROTOKOL
    # Kural 3). With every bin's accuracy below 0.9 (near 0.5, virtually
    # certain at this size), the per-bin |acc-0.9| terms all share the same
    # sign and their weighted sum collapses exactly to |0.9 - 0.5| = 0.4.
    rng = np.random.default_rng(9)
    n = 3_000
    y = np.array([1] * (n // 2) + [0] * (n // 2), dtype=float)
    rng.shuffle(y)
    conf = np.full(n, 0.9)

    assert metrics.ece(y, conf) == pytest.approx(0.4, abs=1e-6)


def test_ece_equal_width_binning_would_be_blind_equal_mass_is_not():
    # All 1500 confidences sit inside a single equal-width bin (a narrow
    # window around 0.95); by construction mean(confidence) == mean(accuracy)
    # over the whole window (0.95 vs 1425/1500=0.95), so a single wide bin
    # reports ECE == 0 -- blind to the fact that the last 75 items (highest
    # confidence) are ALL wrong. Equal-mass bins isolate that tail and see it.
    n = 1500
    conf = np.linspace(0.9401, 0.9599, n)
    y = np.array([1] * 1425 + [0] * 75, dtype=float)

    equal_width_single_bin_ece = abs(np.mean(conf) - np.mean(y))
    assert equal_width_single_bin_ece < 0.01  # confirms the construction is blind to equal-width

    assert metrics.ece(y, conf) > 0.05  # equal-mass sees the miscalibrated tail


# --- overconfidence_rate ------------------------------------------------


def test_overconfidence_rate_hand_computed_case():
    # 10 items; 5 have confidence > OVERCONF_THRESHOLD (0.90): indices
    # 0 (0.95, correct), 1 (0.92, wrong), 3 (0.91, wrong), 6 (0.93, wrong),
    # 8 (0.99, correct). 3 of those 5 are wrong -> rate = 3/5 = 0.6.
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1], dtype=float)
    conf = np.array([0.95, 0.92, 0.5, 0.91, 0.3, 0.6, 0.93, 0.2, 0.99, 0.4])

    assert metrics.overconfidence_rate(y, conf) == pytest.approx(0.6)
    assert config.OVERCONF_THRESHOLD == 0.90  # this hand case assumes the config default


# --- brier ---------------------------------------------------------------


def test_brier_hand_computed_case():
    y = np.array([1, 0, 1, 1, 0], dtype=float)
    conf = np.array([0.8, 0.3, 0.6, 0.9, 0.4])
    expected = np.mean((conf - y) ** 2)  # 0.092

    assert metrics.brier(y, conf) == pytest.approx(expected)


# --- bootstrap_ci ----------------------------------------------------


def test_bootstrap_ci_bounds_and_point_is_full_sample_metric_not_bootstrap_mean():
    rng = np.random.default_rng(21)
    n = 300
    conf = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, conf).astype(float)

    lo, point, hi = metrics.bootstrap_ci(y, conf, metrics.ece)

    assert lo < point < hi
    assert point == pytest.approx(metrics.ece(y, conf), abs=1e-12)


def test_bootstrap_ci_forwards_kwargs_to_metric_fn():
    # The sibling study's bootstrap_ci silently dropped kwargs and crashed
    # at analysis time calling it with a metric that takes threshold=.
    rng = np.random.default_rng(22)
    n = 300
    conf = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, conf).astype(float)

    lo, point, hi = metrics.bootstrap_ci(
        y, conf, metrics.overconfidence_rate, threshold=0.9
    )

    assert lo <= point <= hi
    assert point == pytest.approx(
        metrics.overconfidence_rate(y, conf, threshold=0.9), abs=1e-12
    )


# --- clipping ------------------------------------------------------------


def test_cal_slope_and_intercept_are_finite_with_zero_and_one_confidence():
    # Confidence hitting exactly 0.0 or 1.0 makes logit() +/-inf without
    # clipping (SPEC §3: "Clip confidence to [1e-6, 1-1e-6] first").
    y = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0], dtype=float)
    conf = np.array([1.0, 0.0, 0.9, 1.0, 0.0, 0.8, 0.1, 0.0, 1.0, 0.3])

    assert np.isfinite(metrics.cal_slope(y, conf))
    assert np.isfinite(metrics.cal_intercept(y, conf))
