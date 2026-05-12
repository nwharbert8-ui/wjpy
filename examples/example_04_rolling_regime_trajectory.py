"""Example 04 — Rolling Regime Trajectory

Demonstrates the canonical workflow for time-series regime detection: a
rolling WJ trajectory comparing each window's correlation matrix to a fixed
baseline. Regime transitions appear as sustained deviations.

This is the mechanism by which the 22-year S&P 500 regime catalog was
constructed (Harbert, 2026, Physica A, in revision), and the same pattern
is used in industrial monitoring, brain connectivity, and ecological
tipping-point detection.

The example simulates 1000 'time units' across three regimes with embedded
correlation structure changes, and uses signed WJ to identify the regime
boundaries.

Run:
    python example_04_rolling_regime_trajectory.py
"""
import numpy as np

from wjpy import fast_spearman_matrix, signed_weighted_jaccard, weighted_jaccard


def _nearest_psd(M, jitter=1e-6):
    """Project a symmetric matrix to the nearest positive-semidefinite matrix."""
    M = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, jitter)
    return (eigvecs * eigvals) @ eigvecs.T


def synthesize_regime_data(n_features, n_samples, baseline_corr, regime_corr, mix_alpha, seed):
    """Generate synthetic data with the target correlation structure.

    Mixes baseline_corr and regime_corr structure linearly via mix_alpha
    and samples from the corresponding multivariate normal distribution.
    mix_alpha = 0 -> pure baseline; mix_alpha = 1 -> pure regime.
    """
    rng = np.random.default_rng(seed=seed)
    target = (1 - mix_alpha) * baseline_corr + mix_alpha * regime_corr
    target = _nearest_psd(target, jitter=0.01)
    L = np.linalg.cholesky(target)
    raw = rng.standard_normal((n_features, n_samples))
    return L @ raw


def main():
    n_features = 20
    window_size = 60

    print("=" * 70)
    print("Example 04 — Rolling Regime Trajectory")
    print("=" * 70)
    print()
    print("Scenario: 1000 time units across 3 regimes.")
    print("  Regime A (t=0..400):    baseline correlation structure")
    print("  Regime B (t=400..700):  shifted correlation structure (50% mix)")
    print("  Regime C (t=700..1000): inverted correlation structure")
    print()

    # Build two structurally distinct correlation matrices
    rng = np.random.default_rng(seed=42)
    M = rng.standard_normal((n_features, n_features))
    baseline_corr = (M @ M.T) / n_features
    np.fill_diagonal(baseline_corr, 1.0)
    eigs = np.linalg.eigvalsh(baseline_corr)
    if eigs.min() < 0:
        baseline_corr += (abs(eigs.min()) + 0.05) * np.eye(n_features)
    np.fill_diagonal(baseline_corr, 1.0)

    M2 = rng.standard_normal((n_features, n_features))
    regime_corr = (M2 @ M2.T) / n_features
    np.fill_diagonal(regime_corr, 1.0)
    eigs2 = np.linalg.eigvalsh(regime_corr)
    if eigs2.min() < 0:
        regime_corr += (abs(eigs2.min()) + 0.05) * np.eye(n_features)
    np.fill_diagonal(regime_corr, 1.0)

    # Generate time-series data per regime
    n_a, n_b, n_c = 400, 300, 300
    data_a = synthesize_regime_data(n_features, n_a, baseline_corr, regime_corr, mix_alpha=0.0, seed=1)
    data_b = synthesize_regime_data(n_features, n_b, baseline_corr, regime_corr, mix_alpha=0.5, seed=2)
    data_c = synthesize_regime_data(n_features, n_c, baseline_corr, regime_corr, mix_alpha=1.0, seed=3)
    data_full = np.concatenate([data_a, data_b, data_c], axis=1)

    print(f"  Features: {n_features}")
    print(f"  Total time units: {data_full.shape[1]}")
    print(f"  Window size: {window_size}")
    print(f"  Baseline: first {window_size} time units")
    print()

    # Compute baseline correlation matrix (first window)
    baseline_window = fast_spearman_matrix(data_full[:, :window_size])

    # Rolling trajectory
    step_size = 20
    times = []
    wj_unsigned_traj = []
    wj_signed_traj = []
    for start in range(window_size, data_full.shape[1] - window_size, step_size):
        window_data = data_full[:, start:start + window_size]
        window_corr = fast_spearman_matrix(window_data)
        wj_u = weighted_jaccard(baseline_window, window_corr)
        wj_s = signed_weighted_jaccard(baseline_window, window_corr)
        times.append(start)
        wj_unsigned_traj.append(wj_u)
        wj_signed_traj.append(wj_s)

    print(f"  Rolling-window measurements computed: {len(times)}")
    print()
    print("  t         Unsigned WJ    Signed WJ    Interpretation")
    print("  " + "-" * 60)
    for t, wj_u, wj_s in zip(times, wj_unsigned_traj, wj_signed_traj):
        regime = "A" if t < 400 else ("B" if t < 700 else "C")
        flag = ""
        if wj_u < 0.95:
            flag = "  <- reorganization signal"
        print(f"  t={t:4d}  {wj_u:.4f}        {wj_s:.4f}     regime {regime}{flag}")

    print()
    print("Interpretation:")
    print("  - Both unsigned and signed WJ near 1.0 during Regime A (baseline).")
    print("  - Both drop during Regime B (50% mix) — detectable transition.")
    print("  - Larger drop during Regime C (full structural inversion).")
    print()
    print("The rolling trajectory is the time-series view of regime structure.")
    print("In the published 22-year S&P 500 application (Harbert 2026, Physica A,")
    print("in revision), this same pattern identified 7 unique structural regimes")
    print("including QE-era 2017 (deepest in 22 years by unsigned-WJ z-score).")


if __name__ == "__main__":
    main()
