# Changelog

All notable changes to `wjpy` are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-05-12

### Fixed
- **`implementation_divergence` decomposition was inverted.** On a pure sign
  inversion (`r = +0.8 → r = -0.8`), the unsigned Weighted Jaccard is 1.0 by
  design (blind to sign), which made the previous normalization divide by zero
  and report `sign_inversion_pct = 0.0` / `magnitude_change_pct = 100.0` — the
  opposite of correct. Mixed cases produced negative percentages and values
  above 100%. The decomposition is now built from two non-negative pieces —
  `magnitude_dissimilarity = max(0, 1 - wj_unsigned)` and
  `sign_dissimilarity = max(0, wj_unsigned - wj_signed)` — normalized to their
  sum. A pure sign inversion now correctly reports 100% sign-driven; a pure
  magnitude change reports 100% magnitude-driven; both percentages lie in
  `[0, 100]` and sum to 100 whenever any reorganization is present (and are
  both 0.0 only for identical matrices). `sign_inversion_pct` is documented as
  a conservative estimate (it is 0 when magnitude changes dominate the gap).
- Updated `tests/test_core.py`: `test_pure_sign_inversion_decomposed_as_pure_sign`
  now asserts the corrected 100%/0% split (previously it asserted only `gap < 0`,
  side-stepping the bug). Added `test_pure_magnitude_change_decomposed_as_pure_magnitude`,
  `test_divergence_components_zero_for_identical_matrices`, and
  `test_divergence_percentages_in_unit_range`. `test_divergence_components_sum_to_100`
  is now unconditional for random matrix pairs.
- The `implementation_divergence` docstring and doctests now reflect the
  corrected behavior.

## [0.2.0] - 2026-05-11

### Added
- Type hints throughout `wjpy.core` using `numpy.typing` and standard typing.
- Comprehensive input validation: shape checks, dimensionality checks, NaN
  detection, and informative `ValueError` messages on bad input.
- `pyproject.toml` for modern PEP 517/518 packaging (alongside `setup.py`
  for backward compatibility).
- `examples/` directory with four self-contained runnable scripts:
  - `example_01_quickstart.py` — basic usage of all seven core functions.
  - `example_02_sign_inversion_detection.py` — demonstrates the central
    insight that signed WJ detects sign-inversion regimes that unsigned WJ
    misses (the mechanism behind the 2022-07 financial market regime
    finding cited in Harbert 2026, Physica A, in revision).
  - `example_03_pairing_family_type1_gap.py` — demonstrates the Type 1
    continuous-discrete dissociation gap mechanism (Harbert 2026,
    Frontiers in Pharmacology, in press, doi: 10.3389/fphar.2026.1830847).
  - `example_04_rolling_regime_trajectory.py` — canonical time-series
    workflow for regime detection.
- Expanded test suite: 39 tests across 6 test classes (`TestIdentityAndBaseline`,
  `TestKnownAnswers`, `TestBoundsAndRange`, `TestReferenceParity`,
  `TestInputValidation`, `TestCrossFunctionConsistency`) replacing the
  previous 9 smoke tests.
- Parametrized tests using `pytest.mark.parametrize` for bound checking
  across multiple seeds and threshold values.
- Known-answer tests with hand-computed expected values (proving
  mathematical correctness, not just non-crash behavior).
- GitHub Actions CI configuration that runs tests on every push/PR.

### Changed
- `fast_spearman_matrix` now vectorizes the per-row ranking using
  `numpy.apply_along_axis` (no Python for loop).
- All functions now coerce inputs to `np.float64` for numerical stability.
- `binary_jaccard` now validates that `threshold` is in `(0, 1)`.
- `implementation_divergence` now clips `sign_inversion_pct` to `[-100, 100]`
  to prevent edge-case overflow.
- Diagonal of correlation matrices returned by `fast_spearman_matrix` and
  `fast_pearson_matrix` is now explicitly set to 1.0 (was approximately 1.0
  due to floating-point accumulation).

### Fixed
- `weighted_jaccard_chunked` now correctly handles 1-D input vectors and
  raises clear errors on shape mismatch.
- Removed potential silent failure when correlation matrices are returned
  with non-exact-1.0 diagonals due to floating-point arithmetic.

## [0.1.0] - 2026-05-08

### Added
- Initial release with seven core functions:
  - `weighted_jaccard`
  - `signed_weighted_jaccard`
  - `binary_jaccard`
  - `implementation_divergence`
  - `fast_spearman_matrix`
  - `fast_pearson_matrix`
  - `weighted_jaccard_chunked`
- Smoke test suite (9 tests).
- MIT license.
- PyPI publication.
- Zenodo DOI: 10.5281/zenodo.19025536.
