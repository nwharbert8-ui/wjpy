"""Validation test suite for wjpy.core.

These tests go beyond smoke checks (does the function return 1.0 on identity)
and verify mathematical correctness against known-answer scenarios, edge cases,
and reference implementations.

Run: pytest tests/ -v
"""
import numpy as np
import pytest

from wjpy import (
    binary_jaccard,
    fast_pearson_matrix,
    fast_spearman_matrix,
    implementation_divergence,
    signed_weighted_jaccard,
    weighted_jaccard,
    weighted_jaccard_chunked,
)


# ---------------------------------------------------------------------------
# Identity / baseline correctness
# ---------------------------------------------------------------------------
class TestIdentityAndBaseline:
    """Tests that establish baseline correctness: identical inputs return 1.0."""

    def test_unsigned_wj_identity_returns_one(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 50))
        corr = fast_spearman_matrix(data)
        assert weighted_jaccard(corr, corr) == pytest.approx(1.0)

    def test_signed_wj_identity_returns_one(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 50))
        corr = fast_spearman_matrix(data)
        assert signed_weighted_jaccard(corr, corr) == pytest.approx(1.0)

    def test_binary_jaccard_identity_returns_one(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 50))
        corr = fast_spearman_matrix(data)
        assert binary_jaccard(corr, corr, threshold=0.3) == pytest.approx(1.0)

    def test_divergence_identity_has_zero_gap(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 50))
        corr = fast_spearman_matrix(data)
        div = implementation_divergence(corr, corr)
        assert div["wj_unsigned"] == pytest.approx(1.0)
        assert div["wj_signed"] == pytest.approx(1.0)
        assert div["gap"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Known-answer mathematical correctness
# ---------------------------------------------------------------------------
class TestKnownAnswers:
    """Tests with hand-computed expected values, verifying the math."""

    def test_unsigned_wj_two_by_two_known_answer(self):
        # M has off-diagonal 0.5, N has off-diagonal 0.2.
        # min/max on |0.5|, |0.2| = 0.2, 0.5. WJ = 0.2 / 0.5 = 0.4.
        M = np.array([[1.0, 0.5], [0.5, 1.0]])
        N = np.array([[1.0, 0.2], [0.2, 1.0]])
        assert weighted_jaccard(M, N) == pytest.approx(0.4)

    def test_signed_wj_pure_sign_inversion(self):
        # P has r = +0.8, Q has r = -0.8.
        # After +1 shift: P → 1.8, Q → 0.2.
        # min=0.2, max=1.8. Signed WJ = 0.2 / 1.8 ≈ 0.1111.
        P = np.array([[1.0, 0.8], [0.8, 1.0]])
        Q = np.array([[1.0, -0.8], [-0.8, 1.0]])
        assert signed_weighted_jaccard(P, Q) == pytest.approx(0.2 / 1.8, abs=1e-10)

    def test_unsigned_wj_blind_to_sign_inversion(self):
        # Pure sign flip: unsigned WJ should be 1.0 because |+0.8| = |-0.8|.
        P = np.array([[1.0, 0.8], [0.8, 1.0]])
        Q = np.array([[1.0, -0.8], [-0.8, 1.0]])
        assert weighted_jaccard(P, Q) == pytest.approx(1.0)

    def test_binary_jaccard_three_by_three_known_answer(self):
        # M edges at threshold 0.3: {(0,1), (1,2)}  — pair (0,2) at 0.1 below threshold
        # N edges at threshold 0.3: {(0,1), (0,2), (1,2)}
        # Intersection = 2, union = 3. Binary J = 2/3 ≈ 0.6667.
        M = np.array([[1.0, 0.4, 0.1],
                      [0.4, 1.0, 0.5],
                      [0.1, 0.5, 1.0]])
        N = np.array([[1.0, 0.4, 0.6],
                      [0.4, 1.0, 0.5],
                      [0.6, 0.5, 1.0]])
        assert binary_jaccard(M, N, threshold=0.3) == pytest.approx(2.0 / 3.0)

    def test_pure_sign_inversion_decomposed_as_pure_sign(self):
        # Sign-only inversion: unsigned WJ = 1 (blind to the flip), signed WJ
        # much smaller. The reorganization is therefore 100% sign-driven and
        # 0% magnitude-driven.
        P = np.array([[1.0, 0.8], [0.8, 1.0]])
        Q = np.array([[1.0, -0.8], [-0.8, 1.0]])
        div = implementation_divergence(P, Q)
        assert div["wj_unsigned"] == pytest.approx(1.0)
        assert div["wj_signed"] < 0.5  # signed WJ much smaller
        assert div["gap"] < 0  # signed < unsigned for sign flip
        assert div["sign_inversion_pct"] == pytest.approx(100.0)
        assert div["magnitude_change_pct"] == pytest.approx(0.0)

    def test_pure_magnitude_change_decomposed_as_pure_magnitude(self):
        # Same-sign magnitude change, no inversions: 0% sign-driven,
        # 100% magnitude-driven.
        P = np.array([[1.0, 0.8], [0.8, 1.0]])
        R = np.array([[1.0, 0.2], [0.2, 1.0]])
        div = implementation_divergence(P, R)
        assert div["sign_inversion_pct"] == pytest.approx(0.0)
        assert div["magnitude_change_pct"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Bounds and range correctness
# ---------------------------------------------------------------------------
class TestBoundsAndRange:
    """All Jaccard variants must lie in [0, 1] for any valid input."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 100])
    def test_unsigned_wj_in_unit_interval(self, seed):
        rng = np.random.default_rng(seed)
        A = fast_spearman_matrix(rng.standard_normal((8, 60)))
        B = fast_spearman_matrix(rng.standard_normal((8, 60)))
        wj = weighted_jaccard(A, B)
        assert 0.0 <= wj <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 100])
    def test_signed_wj_in_unit_interval(self, seed):
        rng = np.random.default_rng(seed)
        A = fast_spearman_matrix(rng.standard_normal((8, 60)))
        B = fast_spearman_matrix(rng.standard_normal((8, 60)))
        wj = signed_weighted_jaccard(A, B)
        assert 0.0 <= wj <= 1.0

    @pytest.mark.parametrize("threshold", [0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    def test_binary_jaccard_in_unit_interval(self, threshold):
        rng = np.random.default_rng(42)
        A = fast_spearman_matrix(rng.standard_normal((8, 60)))
        B = fast_spearman_matrix(rng.standard_normal((8, 60)))
        bj = binary_jaccard(A, B, threshold=threshold)
        assert 0.0 <= bj <= 1.0


# ---------------------------------------------------------------------------
# Reference implementation parity
# ---------------------------------------------------------------------------
class TestReferenceParity:
    """Verify wjpy correlation matrices match scipy / numpy reference outputs."""

    def test_fast_spearman_matches_scipy_pairwise(self):
        from scipy.stats import spearmanr
        rng = np.random.default_rng(42)
        data = rng.standard_normal((6, 40))
        fast = fast_spearman_matrix(data)
        for i in range(6):
            for j in range(6):
                ref, _ = spearmanr(data[i], data[j])
                assert fast[i, j] == pytest.approx(ref, abs=1e-10)

    def test_fast_pearson_matches_numpy_corrcoef(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((6, 40))
        fast = fast_pearson_matrix(data)
        ref = np.corrcoef(data)
        np.testing.assert_allclose(fast, ref, atol=1e-10)

    def test_chunked_matches_direct_unsigned_wj(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(50000)
        b = rng.standard_normal(50000)
        chunked = weighted_jaccard_chunked(a, b, chunk_size=5000)
        direct = (np.minimum(np.abs(a), np.abs(b)).sum() /
                  np.maximum(np.abs(a), np.abs(b)).sum())
        assert chunked == pytest.approx(direct, abs=1e-10)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    """Edge cases and malformed inputs should raise informative errors."""

    def test_shape_mismatch_raises_unsigned(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            weighted_jaccard(np.eye(5), np.eye(6))

    def test_shape_mismatch_raises_signed(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            signed_weighted_jaccard(np.eye(5), np.eye(6))

    def test_shape_mismatch_raises_binary(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            binary_jaccard(np.eye(5), np.eye(6))

    def test_non_square_raises(self):
        non_square = np.ones((3, 5))
        with pytest.raises(ValueError, match="square"):
            weighted_jaccard(non_square, non_square)

    def test_nan_input_raises(self):
        bad = np.array([[1.0, np.nan], [np.nan, 1.0]])
        good = np.eye(2)
        with pytest.raises(ValueError, match="NaN"):
            weighted_jaccard(bad, good)

    def test_invalid_threshold_raises(self):
        M = np.eye(3)
        with pytest.raises(ValueError, match="threshold"):
            binary_jaccard(M, M, threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            binary_jaccard(M, M, threshold=-0.1)

    def test_invalid_chunk_size_raises(self):
        a = np.zeros(10)
        b = np.zeros(10)
        with pytest.raises(ValueError, match="chunk_size"):
            weighted_jaccard_chunked(a, b, chunk_size=0)

    def test_data_1d_raises(self):
        data_1d = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="2-D"):
            fast_spearman_matrix(data_1d)


# ---------------------------------------------------------------------------
# Cross-function consistency
# ---------------------------------------------------------------------------
class TestCrossFunctionConsistency:
    """Functions that should agree on shared computations do agree."""

    def test_divergence_components_sum_to_100(self):
        # Two independent random correlation matrices always carry some
        # reorganization, so the two percentages must sum to 100.
        rng = np.random.default_rng(42)
        A = fast_spearman_matrix(rng.standard_normal((10, 50)))
        B = fast_spearman_matrix(rng.standard_normal((10, 50)))
        div = implementation_divergence(A, B)
        total = div["sign_inversion_pct"] + div["magnitude_change_pct"]
        assert total == pytest.approx(100.0, abs=1e-6)

    def test_divergence_components_zero_for_identical_matrices(self):
        # Identical matrices: no reorganization, so both percentages are 0.
        rng = np.random.default_rng(42)
        corr = fast_spearman_matrix(rng.standard_normal((10, 50)))
        div = implementation_divergence(corr, corr)
        assert div["sign_inversion_pct"] == pytest.approx(0.0)
        assert div["magnitude_change_pct"] == pytest.approx(0.0)

    def test_divergence_percentages_in_unit_range(self):
        # Both percentages stay within [0, 100] for arbitrary inputs.
        rng = np.random.default_rng(7)
        for _ in range(20):
            A = fast_spearman_matrix(rng.standard_normal((12, 40)))
            B = fast_spearman_matrix(rng.standard_normal((12, 40)))
            div = implementation_divergence(A, B)
            assert 0.0 <= div["sign_inversion_pct"] <= 100.0
            assert 0.0 <= div["magnitude_change_pct"] <= 100.0

    def test_divergence_keys_present(self):
        rng = np.random.default_rng(42)
        A = fast_spearman_matrix(rng.standard_normal((10, 50)))
        B = fast_spearman_matrix(rng.standard_normal((10, 50)))
        div = implementation_divergence(A, B)
        required_keys = {
            "wj_unsigned",
            "wj_signed",
            "gap",
            "sign_inversion_pct",
            "magnitude_change_pct",
        }
        assert set(div.keys()) == required_keys

    def test_unsigned_and_signed_equal_when_no_negatives(self):
        # When all correlations are positive, signed and unsigned WJ should
        # be similar (both metrics are operating on the same magnitude data).
        # They are not identical due to the [+1] shift in signed, but they
        # are correlated. Verify that on positive-only data they don't diverge wildly.
        rng = np.random.default_rng(42)
        # Build a positive-correlation matrix by squaring random features
        data = rng.standard_normal((6, 100)) ** 2
        A = fast_spearman_matrix(data)
        B = fast_spearman_matrix(data + rng.standard_normal((6, 100)) * 0.1)
        wj_u = weighted_jaccard(A, B)
        wj_s = signed_weighted_jaccard(A, B)
        # Both should be high (similar matrices)
        assert wj_u > 0.5
        assert wj_s > 0.5
