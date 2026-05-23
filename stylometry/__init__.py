"""
stylometry-python
~~~~~~~~~~~~~~~~~

Authorship attribution, stylometric analysis, and LLM detection in Python.

Basic usage::

    from stylometry import StyleAnalyzer           # prose / NLP
    from stylometry import CodeAnalyzer, StyleProfile  # source code
    from stylometry.stats import bootstrap_ci, detect_change_points

:license: MIT
"""

from .analyzer import StyleAnalyzer
from .languages import FUNCTION_WORDS_FR, FUNCTION_WORDS_EN
from .code import CodeAnalyzer, StyleProfile, CODE_FEATURES, SUPPORTED_LANGUAGES
from .stats import (
    bootstrap_ci,
    permutation_test,
    pairwise_tests,
    intra_variance,
    detect_change_points,
    compute_drift,
)
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stylometry-python")
except PackageNotFoundError:
    __version__ = "0.0.0"

__author__ = "Riad Maouchi"
__all__ = [
    # Prose analysis
    "StyleAnalyzer",
    "FUNCTION_WORDS_FR",
    "FUNCTION_WORDS_EN",
    # Code analysis
    "CodeAnalyzer",
    "StyleProfile",
    "CODE_FEATURES",
    "SUPPORTED_LANGUAGES",
    # Statistics
    "bootstrap_ci",
    "permutation_test",
    "pairwise_tests",
    "intra_variance",
    "detect_change_points",
    "compute_drift",
]
