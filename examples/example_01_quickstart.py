"""Example 01 — wjpy Quickstart

Demonstrates the seven core functions of wjpy on synthetic data.

Run:
    pip install wjpy
    python example_01_quickstart.py
"""
import numpy as np

from wjpy import (
    binary_jaccard,
    fast_pearson_matrix,
    fast_spearman_matrix,
    implementation_divergence,
    signed_weighted_jaccard,
    weighted_jaccard,
)


def main():
    rng = np.random.default_rng(seed=42)
    n_features = 50
    n_samples = 200

    # Two simulated observation matrices: baseline regime and stressed regime
    print("=" * 70)
    print("wjpy Quickstart — comparing two correlation architectures")
    print("=" * 70)
    print(f"  Features: {n_features}")
    print(f"  Samples per regime: {n_samples}")
    print(f"  Random seed: 42")
    print()

    data_baseline = rng.standard_normal((n_features, n_samples))
    data_stressed = rng.standard_normal((n_features, n_samples))

    # Compute correlation matrices
    corr_baseline = fast_spearman_matrix(data_baseline)
    corr_stressed = fast_spearman_matrix(data_stressed)

    # 1. Unsigned Weighted Jaccard
    wj_u = weighted_jaccard(corr_baseline, corr_stressed)
    print(f"  Unsigned Weighted Jaccard:    {wj_u:.4f}")
    print(f"    (1.0 = identical magnitude architectures, blind to sign)")

    # 2. Signed Weighted Jaccard
    wj_s = signed_weighted_jaccard(corr_baseline, corr_stressed)
    print(f"  Signed Weighted Jaccard:      {wj_s:.4f}")
    print(f"    (captures sign inversions that unsigned WJ misses)")

    # 3. Binary Jaccard at top-percentile threshold
    bj = binary_jaccard(corr_baseline, corr_stressed, threshold=0.3)
    print(f"  Binary Jaccard (|r| >= 0.3):  {bj:.4f}")
    print(f"    (topological reorganization: edges gained / lost)")
    print()

    # 4. Decomposition
    div = implementation_divergence(corr_baseline, corr_stressed)
    print("  Pairing-family Type 2 decomposition:")
    print(f"    Magnitude-driven reorganization: {div['magnitude_change_pct']:.1f}%")
    print(f"    Sign-driven reorganization:      {div['sign_inversion_pct']:.1f}%")
    print()

    # 5. Pearson comparison
    pearson_baseline = fast_pearson_matrix(data_baseline)
    pearson_stressed = fast_pearson_matrix(data_stressed)
    wj_pearson = weighted_jaccard(pearson_baseline, pearson_stressed)
    print(f"  Same comparison via Pearson:  {wj_pearson:.4f}")
    print(f"    (Type 6 substrate-projection: Spearman vs Pearson gap = "
          f"{abs(wj_u - wj_pearson):.4f})")
    print()
    print("Quickstart complete. See example_02 for sign-inversion detection.")


if __name__ == "__main__":
    main()
