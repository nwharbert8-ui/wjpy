"""Example 02 — Sign-Inversion Detection

Demonstrates the central insight that signed Weighted Jaccard detects
correlation reorganization that unsigned WJ is blind to. This is the
mechanism by which the 2022-07 financial market regime (sign-inversion-driven)
was identified — a finding that single-number correlation methods missed.

The simulated scenario: 30 'assets', baseline regime with mostly-positive
co-movement, stressed regime where half the asset pairs flip sign while
preserving magnitude. The unsigned WJ scores this as near-zero change.
The signed WJ correctly registers it as substantial reorganization.

Run:
    python example_02_sign_inversion_detection.py
"""
import numpy as np

from wjpy import (
    implementation_divergence,
    signed_weighted_jaccard,
    weighted_jaccard,
)


def build_correlation_matrix(n: int, base_corr: float, rng) -> np.ndarray:
    """Construct a synthetic correlation matrix with a known structure."""
    C = np.full((n, n), base_corr) + rng.normal(0, 0.05, size=(n, n))
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.0)
    np.clip(C, -1, 1, out=C)
    return C


def main():
    rng = np.random.default_rng(seed=42)
    n = 30

    print("=" * 70)
    print("Example 02 — Sign-Inversion Detection")
    print("=" * 70)
    print()
    print("Scenario: a financial-market-like regime transition where half")
    print("the asset-pair correlations FLIP SIGN while preserving magnitude.")
    print("Single-number correlation methods score this as near-zero change.")
    print("Signed WJ correctly identifies it as substantial reorganization.")
    print()

    # Baseline: mostly-positive co-movement
    baseline = build_correlation_matrix(n, base_corr=0.4, rng=rng)

    # Stressed regime: take baseline and flip the sign of half the off-diagonal pairs
    stressed = baseline.copy()
    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])
    flip_indices = rng.choice(n_pairs, size=n_pairs // 2, replace=False)
    for k in flip_indices:
        i, j = iu[0][k], iu[1][k]
        stressed[i, j] = -stressed[i, j]
        stressed[j, i] = -stressed[j, i]

    # Compute both WJ variants
    wj_u = weighted_jaccard(baseline, stressed)
    wj_s = signed_weighted_jaccard(baseline, stressed)
    div = implementation_divergence(baseline, stressed)

    print(f"  Number of asset pairs:          {n_pairs}")
    print(f"  Pairs sign-flipped:             {len(flip_indices)} "
          f"({100*len(flip_indices)/n_pairs:.0f}% of all pairs)")
    print()
    print("Result:")
    print(f"  Unsigned WJ:                    {wj_u:.4f}")
    print(f"    (close to 1.0 — unsigned WJ is BLIND to sign inversions)")
    print()
    print(f"  Signed WJ:                      {wj_s:.4f}")
    print(f"    (much lower — signed WJ CORRECTLY detects the reorganization)")
    print()
    print(f"  Type 2 gap (signed - unsigned): {div['gap']:+.4f}")
    print(f"  Sign-driven reorganization:     {div['sign_inversion_pct']:.1f}%")
    print()
    print("This is the methodology that detected the 2022-07 financial market")
    print("regime documented in Harbert (2026, Physica A, in revision).")
    print()
    print("Without signed WJ, the regime is invisible. The unsigned WJ of ~1.0")
    print("means 'no change' — a regime that lasted nearly a month and produced")
    print("large macro stress would be missed by single-number correlation methods.")


if __name__ == "__main__":
    main()
