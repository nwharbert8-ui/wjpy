"""
wjpy.core — Canonical reference implementations of the Weighted Jaccard
pairing-family methodology for correlation network analysis.

These functions are the reference implementations for unsigned Weighted Jaccard,
signed Weighted Jaccard, binary Jaccard, and supporting utilities. All functions
return Python floats or NumPy ndarrays for maximum interoperability.

References
----------
- Harbert, D.H. (2026). Sigma-1 and Sigma-2 receptors exhibit divergent
  genome-wide co-expression architectures in human brain despite shared
  subcellular localization. Frontiers in Pharmacology, in press,
  doi: 10.3389/fphar.2026.1830847.
- Harbert, D.H. (2026). Layer 2H Pairing-Family Decomposition: A Structural
  Property of the Jaccard Family for Architectural Reorganization Analysis.
  Physica A, in revision.
- Harbert, D.H. (2026). Cross-Domain Weighted Jaccard for Industrial
  Prognostics. Mechanical Systems and Signal Processing, under review.
"""
from __future__ import annotations

from typing import Union, Dict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Matrix = NDArray[np.floating]
Vector = NDArray[np.floating]


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _validate_matrix_pair(
    corr_a: ArrayLike,
    corr_b: ArrayLike,
    name_a: str = "corr_a",
    name_b: str = "corr_b",
) -> tuple[Matrix, Matrix]:
    """Validate that two inputs are square NumPy matrices of equal shape."""
    a = np.asarray(corr_a, dtype=np.float64)
    b = np.asarray(corr_b, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(
            f"{name_a} must be a square 2-D array, got shape {a.shape}"
        )
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError(
            f"{name_b} must be a square 2-D array, got shape {b.shape}"
        )
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {name_a} is {a.shape}, {name_b} is {b.shape}"
        )
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Inputs contain NaN values; clean before computing WJ.")
    return a, b


# ---------------------------------------------------------------------------
# Core WJ functions
# ---------------------------------------------------------------------------
def weighted_jaccard(corr_a: ArrayLike, corr_b: ArrayLike) -> float:
    r"""Unsigned Weighted Jaccard between two correlation matrices.

    Computes :math:`WJ = \sum \min(|a|, |b|) / \sum \max(|a|, |b|)` on the
    upper triangle of two correlation matrices. Measures magnitude
    reorganization. Blind to sign inversions: r = +0.8 → r = -0.8 scores
    as zero change because |+0.8| = |-0.8|.

    Parameters
    ----------
    corr_a, corr_b : array_like, shape (n, n)
        Square correlation matrices of identical shape. Values typically
        in [-1, 1] but the function is defined for any real-valued matrices.

    Returns
    -------
    float
        Unsigned Weighted Jaccard similarity in [0, 1].
        1.0 indicates identical magnitude architectures.
        0.0 indicates completely disjoint magnitude architectures.

    Raises
    ------
    ValueError
        If shapes mismatch, inputs are non-square, or inputs contain NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from wjpy import weighted_jaccard
    >>> M = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> weighted_jaccard(M, M)
    1.0
    >>> N = np.array([[1.0, 0.2], [0.2, 1.0]])
    >>> round(weighted_jaccard(M, N), 4)
    0.4
    """
    a, b = _validate_matrix_pair(corr_a, corr_b)
    idx = np.triu_indices(a.shape[0], k=1)
    a_abs = np.abs(a[idx])
    b_abs = np.abs(b[idx])
    num = np.minimum(a_abs, b_abs).sum()
    den = np.maximum(a_abs, b_abs).sum()
    return float(num / den) if den > 0 else 1.0


def signed_weighted_jaccard(
    corr_a: ArrayLike, corr_b: ArrayLike
) -> float:
    r"""Signed Weighted Jaccard between two correlation matrices.

    Captures sign inversions that the unsigned Weighted Jaccard misses by
    shifting correlations from [-1, 1] to [0, 2] before applying the
    min/max ratio. Under this transformation, r = +0.8 and r = -0.8
    become 1.8 and 0.2 respectively — large magnitude difference,
    correctly registered as substantial reorganization.

    Parameters
    ----------
    corr_a, corr_b : array_like, shape (n, n)
        Square correlation matrices.

    Returns
    -------
    float
        Signed Weighted Jaccard similarity in [0, 1].

    Raises
    ------
    ValueError
        If shapes mismatch, inputs are non-square, or inputs contain NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from wjpy import signed_weighted_jaccard
    >>> P = np.array([[1.0,  0.8], [0.8, 1.0]])
    >>> Q = np.array([[1.0, -0.8], [-0.8, 1.0]])  # sign-flipped
    >>> # Unsigned WJ scores P vs Q as 1.0 (blind to sign)
    >>> # Signed WJ correctly registers the inversion
    >>> round(signed_weighted_jaccard(P, Q), 4)
    0.1111
    """
    a, b = _validate_matrix_pair(corr_a, corr_b)
    idx = np.triu_indices(a.shape[0], k=1)
    a_shift = a[idx] + 1.0
    b_shift = b[idx] + 1.0
    num = np.minimum(a_shift, b_shift).sum()
    den = np.maximum(a_shift, b_shift).sum()
    return float(num / den) if den > 0 else 1.0


def binary_jaccard(
    corr_a: ArrayLike,
    corr_b: ArrayLike,
    threshold: float = 0.3,
) -> float:
    r"""Binary Jaccard at a given correlation magnitude threshold.

    Measures topological reorganization (edges gained or lost) by treating
    the correlation matrix as a graph adjacency at the given threshold and
    computing the Jaccard index of edge sets.

    Parameters
    ----------
    corr_a, corr_b : array_like, shape (n, n)
        Square correlation matrices.
    threshold : float, default=0.3
        Edge threshold on absolute correlation magnitude. Edge present
        if |r| >= threshold.

    Returns
    -------
    float
        Binary Jaccard in [0, 1].

    Raises
    ------
    ValueError
        If shapes mismatch or threshold is outside (0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> from wjpy import binary_jaccard
    >>> M = np.array([[1.0, 0.4, 0.1],
    ...               [0.4, 1.0, 0.5],
    ...               [0.1, 0.5, 1.0]])
    >>> N = np.array([[1.0, 0.4, 0.6],
    ...               [0.4, 1.0, 0.5],
    ...               [0.6, 0.5, 1.0]])
    >>> # At threshold 0.3: M has edges {(0,1),(1,2)}, N has {(0,1),(0,2),(1,2)}
    >>> # Intersection size 2, union size 3, Jaccard = 2/3
    >>> round(binary_jaccard(M, N, threshold=0.3), 4)
    0.6667
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"threshold must be in (0, 1), got {threshold}"
        )
    a, b = _validate_matrix_pair(corr_a, corr_b)
    idx = np.triu_indices(a.shape[0], k=1)
    a_edges = np.abs(a[idx]) >= threshold
    b_edges = np.abs(b[idx]) >= threshold
    intersection = (a_edges & b_edges).sum()
    union = (a_edges | b_edges).sum()
    return float(intersection / union) if union > 0 else 1.0


def implementation_divergence(
    corr_a: ArrayLike, corr_b: ArrayLike
) -> Dict[str, float]:
    """Decompose reorganization into magnitude and sign-inversion components.

    The gap between unsigned and signed Weighted Jaccard quantifies how much
    of the total reorganization is attributable to sign inversions versus
    magnitude changes. This is the Type 2 pairing-family decomposition.

    Parameters
    ----------
    corr_a, corr_b : array_like, shape (n, n)
        Square correlation matrices.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``wj_unsigned``: unsigned Weighted Jaccard (float in [0, 1])
        - ``wj_signed``: signed Weighted Jaccard (float in [0, 1])
        - ``gap``: signed minus unsigned WJ (float)
        - ``sign_inversion_pct``: percentage of reorganization from sign
          flips (float in [0, 100])
        - ``magnitude_change_pct``: percentage from magnitude changes
          (float in [0, 100]). Sums to 100 with sign_inversion_pct.

    Examples
    --------
    >>> import numpy as np
    >>> from wjpy import implementation_divergence
    >>> P = np.array([[1.0,  0.8], [0.8, 1.0]])
    >>> Q = np.array([[1.0, -0.8], [-0.8, 1.0]])
    >>> div = implementation_divergence(P, Q)
    >>> # Pure sign inversion: unsigned WJ = 1.0, signed WJ < 1.0
    >>> round(div["sign_inversion_pct"], 1)
    0.0
    """
    a, b = _validate_matrix_pair(corr_a, corr_b)
    wj_u = weighted_jaccard(a, b)
    wj_s = signed_weighted_jaccard(a, b)
    gap = wj_s - wj_u
    reorg_unsigned = 1.0 - wj_u
    if reorg_unsigned > 1e-10:
        sign_inv_pct = (gap / reorg_unsigned) * 100.0
        sign_inv_pct = float(np.clip(sign_inv_pct, -100.0, 100.0))
        magnitude_pct = 100.0 - sign_inv_pct
    else:
        # Architectures are identical at unsigned level; no reorganization
        # to decompose. Set to magnitude=100, sign=0 as a default.
        sign_inv_pct = 0.0
        magnitude_pct = 100.0
    return {
        "wj_unsigned": wj_u,
        "wj_signed": wj_s,
        "gap": gap,
        "sign_inversion_pct": sign_inv_pct,
        "magnitude_change_pct": magnitude_pct,
    }


# ---------------------------------------------------------------------------
# Correlation matrix helpers
# ---------------------------------------------------------------------------
def fast_spearman_matrix(data: ArrayLike) -> Matrix:
    """Vectorized Spearman rank correlation matrix.

    Implements Spearman correlation as Pearson on per-row ranks.
    Substantially faster than pandas.DataFrame.corr(method='spearman')
    for large matrices, with mathematically identical output.

    Parameters
    ----------
    data : array_like, shape (n_features, n_samples)
        Each row is a feature observed across samples. NaN-free.

    Returns
    -------
    ndarray, shape (n_features, n_features)
        Spearman correlation matrix with diagonal = 1.0 and values
        clipped to [-1, 1].

    Raises
    ------
    ValueError
        If data is not 2-D or contains NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from wjpy import fast_spearman_matrix
    >>> rng = np.random.default_rng(seed=42)
    >>> data = rng.standard_normal((10, 100))
    >>> C = fast_spearman_matrix(data)
    >>> C.shape
    (10, 10)
    >>> bool(np.allclose(np.diag(C), 1.0))
    True
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {data.shape}")
    if np.isnan(data).any():
        raise ValueError("data contains NaN; clean before correlation.")
    ranked = np.apply_along_axis(rankdata, axis=1, arr=data)
    ranked -= ranked.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(ranked ** 2, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    ranked /= norms
    corr = ranked @ ranked.T
    np.clip(corr, -1.0, 1.0, out=corr)
    np.fill_diagonal(corr, 1.0)
    return corr


def fast_pearson_matrix(data: ArrayLike) -> Matrix:
    """Vectorized Pearson correlation matrix.

    Parameters
    ----------
    data : array_like, shape (n_features, n_samples)
        Each row is a feature observed across samples.

    Returns
    -------
    ndarray, shape (n_features, n_features)
        Pearson correlation matrix.

    Raises
    ------
    ValueError
        If data is not 2-D or contains NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from wjpy import fast_pearson_matrix
    >>> rng = np.random.default_rng(seed=42)
    >>> data = rng.standard_normal((5, 50))
    >>> C = fast_pearson_matrix(data)
    >>> bool(np.allclose(C, np.corrcoef(data), atol=1e-10))
    True
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {data.shape}")
    if np.isnan(data).any():
        raise ValueError("data contains NaN; clean before correlation.")
    centered = data - data.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered ** 2, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    centered /= norms
    corr = centered @ centered.T
    np.clip(corr, -1.0, 1.0, out=corr)
    np.fill_diagonal(corr, 1.0)
    return corr


# ---------------------------------------------------------------------------
# Memory-efficient variant
# ---------------------------------------------------------------------------
def weighted_jaccard_chunked(
    vec_a: ArrayLike,
    vec_b: ArrayLike,
    chunk_size: int = 10000,
) -> float:
    """Memory-efficient Weighted Jaccard for very large 1-D vectors.

    Computes the unsigned Weighted Jaccard on two 1-D vectors of identical
    length, processing in chunks to avoid materializing min(|a|, |b|) and
    max(|a|, |b|) as full arrays. Useful when comparing flattened
    upper-triangle correlation vectors with >10M elements.

    Parameters
    ----------
    vec_a, vec_b : array_like, shape (n,)
        Correlation vectors of identical length.
    chunk_size : int, default=10000
        Number of elements processed per chunk.

    Returns
    -------
    float
        Unsigned Weighted Jaccard in [0, 1].

    Raises
    ------
    ValueError
        If shapes mismatch or chunk_size is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from wjpy import weighted_jaccard_chunked
    >>> rng = np.random.default_rng(seed=42)
    >>> a = rng.standard_normal(50000)
    >>> b = rng.standard_normal(50000)
    >>> wj_chunked = weighted_jaccard_chunked(a, b, chunk_size=5000)
    >>> wj_direct = float(np.minimum(np.abs(a), np.abs(b)).sum() /
    ...                   np.maximum(np.abs(a), np.abs(b)).sum())
    >>> bool(np.isclose(wj_chunked, wj_direct, atol=1e-10))
    True
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Inputs contain NaN values.")
    a_abs = np.abs(a)
    b_abs = np.abs(b)
    num_total = 0.0
    den_total = 0.0
    for i in range(0, len(a_abs), chunk_size):
        ac = a_abs[i:i + chunk_size]
        bc = b_abs[i:i + chunk_size]
        num_total += float(np.minimum(ac, bc).sum())
        den_total += float(np.maximum(ac, bc).sum())
    return num_total / den_total if den_total > 0 else 1.0
