# Contributing to wjpy

Thank you for your interest in contributing to `wjpy`. This package provides
the canonical reference implementation of the Weighted Jaccard pairing-family
decomposition methodology described in Harbert (2026, doi:
10.3389/fphar.2026.1830847) and related publications.

## How to contribute

### Reporting issues

Open an issue at https://github.com/nwharbert8-ui/wjpy/issues with:
- A clear description of the problem
- A minimal reproducible example (with random seed if applicable)
- The output you observed and what you expected
- Your Python and NumPy/SciPy versions

### Suggesting enhancements

We welcome suggestions for:
- Additional pairing-family decomposition types
- Performance improvements (especially for large matrices)
- Additional substrate-specific examples
- Documentation improvements

Please open an issue before submitting a large pull request so we can
discuss scope.

### Pull requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Make your changes, with tests
5. Verify tests pass: `pytest tests/ -v`
6. Verify linting: `ruff check .`
7. Submit a pull request with a clear description

## Code standards

- **Type hints:** all public functions must have type hints
- **Docstrings:** all public functions must have NumPy-style docstrings with
  Parameters, Returns, Raises, and Examples sections
- **Tests:** new functionality must include validation tests (known-answer
  scenarios), not just smoke tests
- **Input validation:** functions should raise informative `ValueError`s
  on malformed input
- **Reproducibility:** all randomness uses `numpy.random.default_rng(seed=...)`
  for explicit reproducibility

## Mathematical correctness

This package implements peer-reviewed methodology. Any change to a core
mathematical function must:

1. Preserve correctness on existing test cases
2. Include new test cases proving the new behavior
3. Reference the relevant published source or include a derivation comment
4. Pass the `TestReferenceParity` tests (functions matching scipy/numpy
   references must continue to do so)

## Release process

Releases follow [Semantic Versioning](https://semver.org/):
- MAJOR version on incompatible API changes
- MINOR version on backward-compatible new functionality
- PATCH version on backward-compatible bug fixes

Each release updates `CHANGELOG.md`, bumps version in `pyproject.toml`,
and is tagged in the repository.

## Contact

Drake H. Harbert, Inner Architecture LLC
ORCID: 0009-0007-7740-3616
Email: Drake@innerarchitecturellc.com

## License

By contributing, you agree that your contributions will be licensed under
the MIT License (see LICENSE).
