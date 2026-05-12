# wjpy — Weighted Jaccard Pairing-Family Decomposition

[![PyPI version](https://badge.fury.io/py/wjpy.svg)](https://pypi.org/project/wjpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19025536.svg)](https://doi.org/10.5281/zenodo.19025536)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Canonical reference implementation of the Weighted Jaccard pairing-family
decomposition methodology for correlation network analysis. Detects
architectural reorganization in pairwise relationships across two correlation
matrices — including sign-inversion reorganization that single-number
correlation methods miss.

The methodology has been empirically validated across five substrates:
- **Genomics**: Harbert (2026), *Frontiers in Pharmacology* — in press,
  doi: [10.3389/fphar.2026.1830847](https://doi.org/10.3389/fphar.2026.1830847)
- **Industrial sensor networks**: NASA C-MAPSS, Scania APS, Tennessee Eastman
  — Harbert (2026), *MSSP*, under review
- **Financial markets**: 22-year S&P 500 regime catalog — Harbert (2026),
  *Physica A*, in revision
- **Brain connectivity**: propofol-induced reorganization in fMRI — Harbert (2026),
  *Network Neuroscience*, under review
- **Ecological systems**: water quality parameter reorganization — Harbert (2026),
  *Ecological Indicators*, preprint

## Installation

```bash
pip install wjpy
```

Requires Python ≥ 3.8, NumPy ≥ 1.20, SciPy ≥ 1.6.

## Quick start

```python
import numpy as np
from wjpy import (
    fast_spearman_matrix,
    weighted_jaccard,
    signed_weighted_jaccard,
    binary_jaccard,
    implementation_divergence,
)

# Two regimes of observation data
rng = np.random.default_rng(seed=42)
data_baseline = rng.standard_normal((50, 200))
data_stressed = rng.standard_normal((50, 200))

# Compute correlation matrices
corr_baseline = fast_spearman_matrix(data_baseline)
corr_stressed = fast_spearman_matrix(data_stressed)

# Compare network architectures
wj_unsigned = weighted_jaccard(corr_baseline, corr_stressed)
wj_signed   = signed_weighted_jaccard(corr_baseline, corr_stressed)
bj          = binary_jaccard(corr_baseline, corr_stressed, threshold=0.3)

print(f"Unsigned WJ: {wj_unsigned:.4f}  (magnitude reorganization)")
print(f"Signed WJ:   {wj_signed:.4f}  (magnitude + sign reorganization)")
print(f"Binary J:    {bj:.4f}  (topological reorganization at |r|>=0.3)")

# Decompose into magnitude vs sign components (Type 2 pairing-family gap)
div = implementation_divergence(corr_baseline, corr_stressed)
print(f"Magnitude-driven: {div['magnitude_change_pct']:.1f}%")
print(f"Sign-driven:      {div['sign_inversion_pct']:.1f}%")
```

## Worked examples

Four self-contained runnable examples are in `examples/`:

| Example | What it demonstrates |
|---|---|
| `example_01_quickstart.py` | Basic usage of all seven core functions |
| `example_02_sign_inversion_detection.py` | Sign-inversion detection (the mechanism behind the 2022-07 financial regime finding that single-number methods miss) |
| `example_03_pairing_family_type1_gap.py` | The Type 1 continuous-discrete dissociation gap mechanism (cited in Harbert 2026, *Frontiers in Pharmacology*, doi: 10.3389/fphar.2026.1830847) |
| `example_04_rolling_regime_trajectory.py` | Canonical time-series workflow for rolling regime detection |

Run any of them after installing the package:

```bash
python examples/example_01_quickstart.py
python examples/example_02_sign_inversion_detection.py
python examples/example_03_pairing_family_type1_gap.py
python examples/example_04_rolling_regime_trajectory.py
```

## API reference

| Function | Purpose |
|---|---|
| `weighted_jaccard(corr_a, corr_b)` | Unsigned WJ — measures magnitude reorganization; blind to sign |
| `signed_weighted_jaccard(corr_a, corr_b)` | Signed WJ — captures sign inversions that unsigned WJ misses |
| `binary_jaccard(corr_a, corr_b, threshold)` | Topological reorganization at edge threshold |
| `implementation_divergence(corr_a, corr_b)` | Decompose reorganization into magnitude vs sign components (Type 2 pairing-family) |
| `fast_spearman_matrix(data)` | Vectorized Spearman correlation matrix |
| `fast_pearson_matrix(data)` | Vectorized Pearson correlation matrix |
| `weighted_jaccard_chunked(vec_a, vec_b)` | Memory-efficient WJ for very large 1-D vectors (>10M elements) |

All functions have full NumPy-style docstrings with Parameters, Returns,
Raises, and runnable Examples sections. Use `help(function_name)` to view.

## Methodology overview

Most correlation-network analysis collapses two structurally distinct
reorganization modes into one number:

- **Magnitude reorganization**: correlations grow or shrink uniformly
  (financial crises, system-wide stress events)
- **Sign reorganization**: correlations flip polarity without changing
  magnitude (calm-era regime transitions, neural circuit polarity changes)

`wjpy` implements both modes — and the gap between them — as separate,
interpretable measurements. The Type 2 pairing-family gap (`signed_WJ −
unsigned_WJ`) quantifies how much of any reorganization is sign-driven
versus magnitude-driven.

The pairing-family decomposition is documented in detail in the methodology
paper currently under revision at *Physica A*, and a worked example with
empirical validation is in `examples/example_03_pairing_family_type1_gap.py`.

## Testing

The test suite includes 39 tests across six test classes:

- **TestIdentityAndBaseline** — identity returns 1.0
- **TestKnownAnswers** — hand-computed expected values prove correctness
- **TestBoundsAndRange** — outputs always in [0, 1]
- **TestReferenceParity** — `fast_spearman` matches scipy; `fast_pearson` matches numpy
- **TestInputValidation** — informative errors on malformed input
- **TestCrossFunctionConsistency** — decomposition components sum to 100%

Run the suite:

```bash
pip install wjpy[test]
pytest tests/ -v
```

## Citing wjpy

If you use `wjpy` in published work, please cite:

```bibtex
@software{harbert_wjpy_2026,
  author       = {Harbert, Drake H.},
  title        = {wjpy: Weighted Jaccard pairing-family decomposition for correlation network analysis},
  year         = 2026,
  version      = {0.2.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19025536},
  url          = {https://github.com/nwharbert8-ui/wjpy}
}
```

And the relevant peer-reviewed source:

```bibtex
@article{harbert_sigma_2026,
  author  = {Harbert, Drake H.},
  title   = {Sigma-1 and Sigma-2 receptors exhibit divergent genome-wide co-expression architectures in human brain despite shared subcellular localization},
  journal = {Frontiers in Pharmacology},
  year    = 2026,
  doi     = {10.3389/fphar.2026.1830847}
}
```

## License

MIT License. See [LICENSE](LICENSE) for the full text.

## Author

**Drake H. Harbert**
Founder, Inner Architecture LLC, Canton, OH, USA
ORCID: [0009-0007-7740-3616](https://orcid.org/0009-0007-7740-3616)
Email: Drake@innerarchitecturellc.com

## See also

- [Cross-domain reference implementation](https://github.com/nwharbert8-ui/financial-wj-structural-regimes) — the canonical pipeline applying `wjpy` to S&P 500 financial market data, with full reproducibility from public data and fixed random seed.
- [WJ Regime](https://wjregime.substack.com) — methodology newsletter applying `wjpy` to live financial market data weekly.
- [Inner Architecture LLC](https://innerarchitecturellc.com) — the institutional vehicle for this research.
