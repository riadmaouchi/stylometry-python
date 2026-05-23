"""
Tests for stylometry.stats.
Run: pytest tests/ -v
"""

import numpy as np
import pytest

from stylometry.stats import (
    bootstrap_ci,
    compute_drift,
    detect_change_points,
    intra_variance,
    pairwise_tests,
    permutation_test,
)


class TestBootstrapCI:

    def test_returns_tuple_of_two_floats(self):
        lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5])
        assert isinstance(lo, float) and isinstance(hi, float)

    def test_lower_less_than_upper(self):
        lo, hi = bootstrap_ci([0.2] * 10 + [0.8] * 10)
        assert lo < hi

    def test_ci_contains_mean(self):
        data = [float(x) for x in range(1, 21)]
        lo, hi = bootstrap_ci(data)
        mean = float(np.mean(data))
        assert lo <= mean <= hi

    def test_reproducible_with_rng(self):
        data = [0.1 * i for i in range(20)]
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        assert bootstrap_ci(data, rng=rng1) == bootstrap_ci(data, rng=rng2)

    def test_custom_confidence_level(self):
        data = list(range(100))
        lo95, hi95 = bootstrap_ci(data, ci=0.95)
        lo50, hi50 = bootstrap_ci(data, ci=0.50)
        assert (hi95 - lo95) > (hi50 - lo50)


class TestPermutationTest:

    def test_returns_dict_with_required_keys(self):
        result = permutation_test([0.1, 0.2], [0.8, 0.9])
        for key in ("observed", "p_value", "significant_05", "significant_01"):
            assert key in result

    def test_p_value_in_range(self):
        result = permutation_test([0.1] * 20, [0.5] * 20)
        assert 0.0 < result["p_value"] <= 1.0

    def test_significant_for_clear_separation(self):
        result = permutation_test([0.0] * 50, [1.0] * 50, n_perm=1000)
        assert result["significant_05"] is True

    def test_not_significant_for_identical(self):
        # Same data in both groups: observed stat = 0, all permutations >= 0 → p = 1.0
        data = [0.5] * 20
        result = permutation_test(data, data, n_perm=100)
        assert result["p_value"] > 0.5

    def test_observed_is_non_negative(self):
        result = permutation_test([0.3, 0.4], [0.7, 0.8])
        assert result["observed"] >= 0.0


class TestPairwiseTests:

    def test_returns_list(self):
        data = {"A": [0.1, 0.2], "B": [0.5, 0.6], "C": [0.9, 0.95]}
        results = pairwise_tests(data)
        assert isinstance(results, list)
        assert len(results) == 3  # C(3,2)

    def test_each_result_has_required_keys(self):
        data = {"A": [0.1, 0.2, 0.3], "B": [0.8, 0.9, 0.95]}
        for r in pairwise_tests(data):
            assert "model_a" in r
            assert "model_b" in r
            assert "p_corrected" in r
            assert "significant_05" in r

    def test_sorted_by_p_corrected(self):
        data = {"A": [0.1] * 20, "B": [0.9] * 20, "C": [0.5] * 20}
        results = pairwise_tests(data)
        p_values = [r["p_corrected"] for r in results]
        assert p_values == sorted(p_values)

    def test_invalid_correction_raises(self):
        with pytest.raises(ValueError, match="correction must be"):
            pairwise_tests({"A": [1.0], "B": [2.0]}, correction="fdr")

    def test_holm_correction(self):
        data = {"A": [0.1] * 10, "B": [0.5] * 10, "C": [0.9] * 10}
        results = pairwise_tests(data, correction="holm")
        assert all("p_corrected" in r for r in results)

    def test_none_correction(self):
        data = {"A": [0.1] * 10, "B": [0.9] * 10}
        results = pairwise_tests(data, correction="none")
        # Without correction, p_corrected == p_raw
        assert abs(results[0]["p_corrected"] - results[0]["p_raw"]) < 1e-9


class TestIntraVariance:

    def test_returns_dict_per_group(self):
        from stylometry import StyleAnalyzer

        # Use min_words=3 to accept short test sentences
        sa = StyleAnalyzer(language="en", min_words=3)
        texts = {
            "A": ["the cat sat on the mat", "a dog ran in the park"],
            "B": ["he went to the store", "she bought some food"],
        }
        result = intra_variance(texts, sa)
        assert set(result.keys()) == {"A", "B"}

    def test_single_text_returns_zero(self):
        from stylometry import StyleAnalyzer

        sa = StyleAnalyzer(language="en", min_words=3)
        result = intra_variance({"A": ["only one text here"]}, sa)
        assert result["A"] == pytest.approx(0.0)


class TestDetectChangePoints:

    def test_detects_large_jump(self):
        series = [10.0, 12.0, 11.0, 13.0, 10.0, 65.0, 68.0, 70.0, 72.0, 69.0]
        results = detect_change_points(series, min_size=2, min_magnitude=20.0)
        assert len(results) > 0
        assert results[0]["magnitude"] > 20.0

    def test_stable_series_no_change_points(self):
        series = [20.0, 21.0, 19.0, 22.0, 20.5, 21.0, 19.5]
        results = detect_change_points(series, min_size=2, min_magnitude=5.0)
        assert len(results) == 0

    def test_result_has_required_keys(self):
        series = [10.0] * 5 + [70.0] * 5
        results = detect_change_points(series, min_size=2, min_magnitude=5.0)
        if results:
            for key in ("index", "value_before", "value_after", "magnitude"):
                assert key in results[0]

    def test_too_short_series_returns_empty(self):
        assert detect_change_points([10.0, 20.0], min_size=3) == []

    def test_sorted_by_magnitude(self):
        series = [10.0] * 3 + [80.0] * 3 + [15.0] * 3
        results = detect_change_points(
            series, min_size=2, penalty=1.0, min_magnitude=5.0
        )
        if len(results) > 1:
            magnitudes = [r["magnitude"] for r in results]
            assert magnitudes == sorted(magnitudes, reverse=True)


class TestComputeDrift:

    def test_returns_drift(self):
        scores = [10.0, 12.0, 11.0, 60.0, 62.0, 65.0]
        result = compute_drift(scores, split_index=3)
        assert "drift" in result
        assert result["drift"] > 0

    def test_drift_positive_when_post_higher(self):
        result = compute_drift([10.0, 10.0, 10.0, 50.0, 50.0, 50.0], split_index=3)
        assert result["drift"] == pytest.approx(40.0, abs=0.1)

    def test_empty_pre_omits_baseline(self):
        result = compute_drift([50.0, 55.0], split_index=0)
        assert "baseline_mean" not in result
        assert "post_mean" in result

    def test_empty_post_omits_drift(self):
        result = compute_drift([10.0, 12.0], split_index=2)
        assert "drift" not in result
        assert "baseline_mean" in result
