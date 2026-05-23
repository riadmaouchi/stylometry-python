"""
stats.py
~~~~~~~~

Statistical tests for stylometric analysis.

Functions
---------
bootstrap_ci()         Percentile bootstrap confidence interval
permutation_test()     Two-sample mean-difference permutation test
pairwise_tests()       All C(n,2) Welch tests with multiple-comparison correction
intra_variance()       Within-group cosine distance (style cohesion baseline)
detect_change_points() PELT change-point detection on a time series
compute_drift()        Pre/post-LLM era style drift summary
"""

from __future__ import annotations

import itertools
from itertools import combinations
from typing import Any, Literal

import numpy as np
from scipy.stats import ttest_ind

# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: list[float] | np.ndarray,
    n_boot: int = 5_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval.

    Parameters
    ----------
    data   : sample of shift values (floats)
    n_boot : number of bootstrap draws
    ci     : confidence level (0.95 → 95% CI)
    rng    : NumPy Generator for reproducibility

    Returns
    -------
    (lower_bound, upper_bound)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray(data, dtype=float)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(boots, alpha * 100)),
        float(np.percentile(boots, (1 - alpha) * 100)),
    )


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def permutation_test(
    group_a: list[float] | np.ndarray,
    group_b: list[float] | np.ndarray,
    n_perm: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Two-sided permutation test on the difference of means.

    H₀: both groups share the same distribution.
    Statistic: |mean(A) − mean(B)|

    Returns
    -------
    dict with keys:
        observed          — observed test statistic
        p_value           — p-value (conservative lower bound: 1/n_perm)
        null_distribution — array of statistics under H₀
        significant_05    — bool (p < 0.05)
        significant_01    — bool (p < 0.01)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    observed = float(abs(a.mean() - b.mean()))
    n_a = len(a)

    combined = np.concatenate([a, b])
    perms = rng.permutation(np.tile(combined, (n_perm, 1)).T).T
    null = np.abs(perms[:, :n_a].mean(axis=1) - perms[:, n_a:].mean(axis=1))

    p_value = float(max((null >= observed).mean(), 1.0 / n_perm))
    return {
        "observed": observed,
        "p_value": p_value,
        "null_distribution": null,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
    }


# ---------------------------------------------------------------------------
# Pairwise tests
# ---------------------------------------------------------------------------

_CORRECTIONS = frozenset({"bonferroni", "holm", "none"})


def pairwise_tests(
    shifts_dict: dict[str, list[float]],
    correction: Literal["bonferroni", "holm", "none"] = "bonferroni",
) -> list[dict[str, Any]]:
    """
    Welch t-tests on all C(n, 2) pairs with multiple-comparison correction.

    Parameters
    ----------
    shifts_dict : {"Group A": [...], "Group B": [...], ...}
    correction  : "bonferroni" | "holm" | "none"

    Returns
    -------
    list of dicts sorted by p_corrected ascending
    """
    if correction not in _CORRECTIONS:
        raise ValueError(
            f"correction must be one of {_CORRECTIONS!r}, got {correction!r}"
        )

    models = list(shifts_dict.keys())
    pairs = list(itertools.combinations(models, 2))
    n_tests = len(pairs)

    raw: list[dict[str, Any]] = []
    for a, b in pairs:
        t, p = ttest_ind(shifts_dict[a], shifts_dict[b], equal_var=False)
        raw.append({"model_a": a, "model_b": b, "t": float(t), "p_raw": float(p)})

    raw.sort(key=lambda r: r["p_raw"])

    for i, r in enumerate(raw):
        if correction == "bonferroni":
            p_corr = min(r["p_raw"] * n_tests, 1.0)
        elif correction == "holm":
            p_corr = min(r["p_raw"] * (n_tests - i), 1.0)
        else:
            p_corr = r["p_raw"]
        r.update(
            {
                "p_corrected": p_corr,
                "significant_05": p_corr < 0.05,
                "correction": correction,
                "n_tests": n_tests,
            }
        )

    return sorted(raw, key=lambda r: r["p_corrected"])


# ---------------------------------------------------------------------------
# Intra-group variance
# ---------------------------------------------------------------------------


def intra_variance(
    texts_dict: dict[str, list[str]],
    analyzer: Any,
) -> dict[str, float]:
    """
    Within-group cosine distance — natural style noise of a group.

    For each group: mean cosine distance between all text pairs.
    A cohesive group (e.g. a single LLM) has low intra-variance.

    Parameters
    ----------
    texts_dict : {"Zola": [...], "GPT-4": [...], ...}
    analyzer   : instance exposing shift(text_a, text_b) → float

    Returns
    -------
    {"Zola": 0.48, "GPT-4": 0.12, ...}
    """
    result: dict[str, float] = {}
    for label, texts in texts_dict.items():
        if len(texts) < 2:
            result[label] = 0.0
            continue
        dists = [analyzer.shift(t1, t2) for t1, t2 in combinations(texts, 2)]
        result[label] = float(np.mean(dists))
    return result


# ---------------------------------------------------------------------------
# Change-point detection
# ---------------------------------------------------------------------------


def detect_change_points(
    series: list[float] | np.ndarray,
    min_size: int = 3,
    penalty: float = 3.0,
    min_magnitude: float = 5.0,
) -> list[dict[str, Any]]:
    """
    Detect stylistic breakpoints in a time series using PELT (ruptures).

    Parameters
    ----------
    series        : ordered sequence of style scores (e.g. quarterly LLM scores)
    min_size      : minimum number of points per segment
    penalty       : PELT penalty for adding a breakpoint (lower = more breakpoints)
    min_magnitude : minimum |before − after| to report a change point

    Returns
    -------
    list of dicts, each with:
        index         — index in series where the break occurs
        value_before  — mean of the segment before the break
        value_after   — mean of the segment after the break
        magnitude     — |value_after − value_before|
    Sorted by magnitude descending.
    """
    try:
        import ruptures as rpt
    except ImportError as e:
        raise ImportError(
            "ruptures is required for change-point detection: pip install ruptures"
        ) from e

    arr = np.asarray(series, dtype=float)
    if len(arr) < min_size * 2:
        return []

    try:
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(arr.reshape(-1, 1))
        breakpoints = algo.predict(pen=penalty)
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for bkp in breakpoints[:-1]:  # last element is len(arr), not a real breakpoint
        if bkp >= len(arr):
            continue
        before = arr[max(0, bkp - min_size) : bkp]
        after = arr[bkp : min(len(arr), bkp + min_size)]
        if len(before) == 0 or len(after) == 0:
            continue
        v_before = float(np.mean(before))
        v_after = float(np.mean(after))
        magnitude = abs(v_after - v_before)
        if magnitude < min_magnitude:
            continue
        results.append(
            {
                "index": bkp,
                "value_before": round(v_before, 2),
                "value_after": round(v_after, 2),
                "magnitude": round(magnitude, 2),
            }
        )

    return sorted(results, key=lambda r: r["magnitude"], reverse=True)


# ---------------------------------------------------------------------------
# Drift summary
# ---------------------------------------------------------------------------


def compute_drift(
    scores: list[float],
    split_index: int,
) -> dict[str, float]:
    """
    Summarize style drift between a pre-event and post-event segment.

    Parameters
    ----------
    scores      : ordered list of style scores
    split_index : index separating pre (scores[:split_index]) from post

    Returns
    -------
    dict with 'baseline_mean', 'post_mean', 'drift', 'baseline_std', 'post_std'
    (keys omitted if the corresponding segment is empty)
    """
    pre = np.asarray(scores[:split_index], dtype=float)
    post = np.asarray(scores[split_index:], dtype=float)
    result: dict[str, float] = {}
    if len(pre) > 0:
        result["baseline_mean"] = round(float(pre.mean()), 2)
        result["baseline_std"] = round(float(pre.std()), 2)
    if len(post) > 0:
        result["post_mean"] = round(float(post.mean()), 2)
        result["post_std"] = round(float(post.std()), 2)
    if "baseline_mean" in result and "post_mean" in result:
        result["drift"] = round(result["post_mean"] - result["baseline_mean"], 2)
    return result
