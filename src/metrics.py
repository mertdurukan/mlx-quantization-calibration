"""ECE, calibration slope/intercept, Brier, overconfidence, bootstrap CI (SPEC §3)."""
import numpy as np
import statsmodels.api as sm

import src.config as config


def _clip_conf(y_conf):
    return np.clip(np.asarray(y_conf, dtype=float), 1e-6, 1 - 1e-6)


def _logit(p):
    return np.log(p / (1 - p))


def ece(y_correct, y_conf):
    """15 equal-MASS bins via np.array_split on confidence-sorted order (SPEC §3)."""
    y_correct = np.asarray(y_correct, dtype=float)
    y_conf = np.asarray(y_conf, dtype=float)
    order = np.argsort(y_conf, kind="stable")
    correct_bins = np.array_split(y_correct[order], config.ECE_N_BINS)
    conf_bins = np.array_split(y_conf[order], config.ECE_N_BINS)
    n_total = len(y_conf)
    total = 0.0
    for correct_bin, conf_bin in zip(correct_bins, conf_bins):
        if len(correct_bin) == 0:
            continue
        total += (len(correct_bin) / n_total) * abs(
            np.mean(correct_bin) - np.mean(conf_bin)
        )
    return total


def cal_slope(y_correct, y_conf):
    """Logistic regression of correctness on logit(confidence); slope coefficient."""
    y_correct = np.asarray(y_correct, dtype=float)
    x = _logit(_clip_conf(y_conf))
    model = sm.GLM(y_correct, sm.add_constant(x), family=sm.families.Binomial())
    return model.fit().params[1]


def cal_intercept(y_correct, y_conf):
    """statsmodels GLM with offset=logit(confidence) (slope fixed to 1); intercept."""
    y_correct = np.asarray(y_correct, dtype=float)
    x = _logit(_clip_conf(y_conf))
    model = sm.GLM(y_correct, np.ones_like(x), offset=x, family=sm.families.Binomial())
    return model.fit().params[0]


def brier(y_correct, y_conf):
    y_correct = np.asarray(y_correct, dtype=float)
    y_conf = np.asarray(y_conf, dtype=float)
    return float(np.mean((y_conf - y_correct) ** 2))


def overconfidence_rate(y_correct, y_conf, threshold=config.OVERCONF_THRESHOLD):
    """Fraction of items with confidence above threshold that are wrong."""
    y_correct = np.asarray(y_correct, dtype=float)
    y_conf = np.asarray(y_conf, dtype=float)
    above = y_conf > threshold
    return float(np.mean(y_correct[above] == 0))


def mean_conf_by_correctness(y_correct, y_conf):
    """-> (mean confidence | incorrect, mean confidence | correct)."""
    y_correct = np.asarray(y_correct, dtype=float)
    y_conf = np.asarray(y_conf, dtype=float)
    is_correct = y_correct == 1
    return float(np.mean(y_conf[~is_correct])), float(np.mean(y_conf[is_correct]))


def bootstrap_ci(y_correct, y_conf, metric_fn, n=config.BOOTSTRAP_N, seed=config.SEED, **metric_kwargs):
    """Percentile bootstrap over items; (lo, point, hi). Point = full-sample metric,
    not the bootstrap mean. **metric_kwargs forwarded to both point and every resample."""
    y_correct = np.asarray(y_correct, dtype=float)
    y_conf = np.asarray(y_conf, dtype=float)
    point = metric_fn(y_correct, y_conf, **metric_kwargs)

    rng = np.random.default_rng(seed)
    n_items = len(y_correct)
    samples = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_items, n_items)
        samples[i] = metric_fn(y_correct[idx], y_conf[idx], **metric_kwargs)

    lo, hi = np.percentile(samples, [config.CI_LOW, config.CI_HIGH])
    return lo, point, hi
