"""
wjpy - Weighted Jaccard methodology for correlation network analysis.

Author: Drake H. Harbert
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616

Core (free) tier:
    weighted_jaccard, signed_weighted_jaccard, binary_jaccard,
    fast_spearman_matrix, fast_pearson_matrix,
    implementation_divergence, weighted_jaccard_chunked.

Premium tier (separate distribution, see PREMIUM.md):
    pairing_family.type1_gap, pairing_family.type2_gap,
    regime_detection.sliding_baseline_detector,
    macro_alignment.align_to_window.

License: MIT (core tier).
Documentation: https://github.com/nwharbert8-ui/wjpy
"""
from .core import (
    weighted_jaccard,
    signed_weighted_jaccard,
    binary_jaccard,
    fast_spearman_matrix,
    fast_pearson_matrix,
    implementation_divergence,
    weighted_jaccard_chunked,
)

__version__ = "0.2.0"
__author__ = "Drake H. Harbert"
__email__ = "Drake@innerarchitecturellc.com"
__license__ = "MIT"

__all__ = [
    "weighted_jaccard",
    "signed_weighted_jaccard",
    "binary_jaccard",
    "fast_spearman_matrix",
    "fast_pearson_matrix",
    "implementation_divergence",
    "weighted_jaccard_chunked",
]
