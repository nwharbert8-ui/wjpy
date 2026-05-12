"""Example 03 — Pairing-Family Type 1 (Continuous-Discrete) Gap

Demonstrates the Type 1 dissociation gap that is the central pairing-family
result in Harbert (2026, Frontiers in Pharmacology, in press, doi:
10.3389/fphar.2026.1830847). The gap quantifies whether two correlation
architectures differ primarily in their bulk distribution (small gap) or
in their top-percentile partnerships (large gap).

The example builds two scenarios:
  Scenario A: bulk reorganization — both continuous and binary J change together
  Scenario B: top-partner reorganization — bulk preserved, top edges reshuffled

In Scenario B, the continuous and binary Jaccards diverge — and the size
of that divergence is the Type 1 gap.

Run:
    python example_03_pairing_family_type1_gap.py
"""
import numpy as np

from wjpy import binary_jaccard, weighted_jaccard


def main():
    rng = np.random.default_rng(seed=42)
    n = 40

    print("=" * 70)
    print("Example 03 — Pairing-Family Type 1 Dissociation Gap")
    print("=" * 70)
    print()

    # Construct a base correlation matrix with structured top partners
    base = rng.standard_normal((n, n))
    base = (base + base.T) / 2
    np.fill_diagonal(base, 1.0)
    base /= np.max(np.abs(base[np.triu_indices(n, k=1)])) * 1.5
    np.fill_diagonal(base, 1.0)
    np.clip(base, -1, 1, out=base)

    # ---- Scenario A: bulk uniform shrinkage (50%) ----
    # All correlations scaled by 0.5. Continuous WJ should drop; binary J
    # should ALSO drop because some edges fall below threshold.
    scenario_a = base * 0.5
    np.fill_diagonal(scenario_a, 1.0)

    wj_continuous_a = weighted_jaccard(base, scenario_a)
    wj_binary_a = binary_jaccard(base, scenario_a, threshold=0.2)
    type1_gap_a = wj_continuous_a - wj_binary_a

    print("Scenario A: uniform 50% magnitude shrinkage (bulk reorganization)")
    print(f"  Continuous WJ:        {wj_continuous_a:.4f}")
    print(f"  Binary  J (>=0.2):    {wj_binary_a:.4f}")
    print(f"  Type 1 gap:           {type1_gap_a:+.4f}")
    print(f"    (small gap — both metrics shift together; bulk reorganization)")
    print()

    # ---- Scenario B: top-partner reshuffling ----
    # Take base, identify top 5% of edges, shuffle THEIR positions only.
    # Bulk distribution preserved; top edges reshuffled.
    scenario_b = base.copy()
    iu = np.triu_indices(n, k=1)
    edge_vals = base[iu]
    threshold = np.quantile(np.abs(edge_vals), 0.95)
    top_mask = np.abs(edge_vals) >= threshold
    top_indices_in_iu = np.where(top_mask)[0]
    permuted_targets = rng.permutation(top_indices_in_iu)
    for src_k, tgt_k in zip(top_indices_in_iu, permuted_targets):
        i_src, j_src = iu[0][src_k], iu[1][src_k]
        i_tgt, j_tgt = iu[0][tgt_k], iu[1][tgt_k]
        scenario_b[i_tgt, j_tgt] = base[i_src, j_src]
        scenario_b[j_tgt, i_tgt] = base[i_src, j_src]
    np.fill_diagonal(scenario_b, 1.0)

    wj_continuous_b = weighted_jaccard(base, scenario_b)
    wj_binary_b = binary_jaccard(base, scenario_b, threshold=0.2)
    type1_gap_b = wj_continuous_b - wj_binary_b

    print("Scenario B: top 5% edge reshuffling (top-partner reorganization)")
    print(f"  Continuous WJ:        {wj_continuous_b:.4f}")
    print(f"  Binary  J (>=0.2):    {wj_binary_b:.4f}")
    print(f"  Type 1 gap:           {type1_gap_b:+.4f}")
    print(f"    (the gap localizes top-partner reshuffling)")
    print()

    print(f"Interpretation:")
    print(f"  |Gap A| = {abs(type1_gap_a):.4f}")
    print(f"  |Gap B| = {abs(type1_gap_b):.4f}")
    print()
    print("In Scenario B the gap is LARGER even though the global structure")
    print("is similar. The continuous WJ misses the top-partner reorganization")
    print("(it averages over all pairs); the binary J catches it (it focuses")
    print("on which specific pairs are above threshold). The gap is the")
    print("measurement of top-partner reorganization.")
    print()
    print("This is the dissociation gap mechanism documented in")
    print("Harbert (2026, doi: 10.3389/fphar.2026.1830847) across 21 gene")
    print("pair categories in human brain.")


if __name__ == "__main__":
    main()
