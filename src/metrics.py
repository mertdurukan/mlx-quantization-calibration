"""Interface skeleton only — SPEC §3 signatures, zero logic (PROTOKOL Kural 3 addendum).

Every body raises NotImplementedError. This exists so tests/test_metrics.py can be
collected and fail per-test for the right reason, instead of one collection-level
ModuleNotFoundError. Filled in by Görev 3 — no test may be weakened to make it pass.
"""
import src.config as config


def ece(y_correct, y_conf):
    """15 equal-MASS bins via np.array_split on confidence-sorted order (SPEC §3)."""
    raise NotImplementedError


def cal_slope(y_correct, y_conf):
    """Logistic regression of correctness on logit(confidence); slope coefficient."""
    raise NotImplementedError


def cal_intercept(y_correct, y_conf):
    """statsmodels GLM with offset=logit(confidence) (slope fixed to 1); intercept."""
    raise NotImplementedError


def brier(y_correct, y_conf):
    raise NotImplementedError


def overconfidence_rate(y_correct, y_conf, threshold=config.OVERCONF_THRESHOLD):
    """Fraction of items with confidence above threshold that are wrong."""
    raise NotImplementedError


def mean_conf_by_correctness(y_correct, y_conf):
    """-> (mean confidence | incorrect, mean confidence | correct)."""
    raise NotImplementedError


def bootstrap_ci(y_correct, y_conf, metric_fn, n=config.BOOTSTRAP_N, seed=config.SEED, **metric_kwargs):
    """Percentile bootstrap over items; (lo, point, hi). Point = full-sample metric,
    not the bootstrap mean. **metric_kwargs forwarded to both point and every resample."""
    raise NotImplementedError
