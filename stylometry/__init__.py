"""
stylometry-python
~~~~~~~~~~~~~~~~~

Authorship attribution and stylometric analysis in Python.

Basic usage::

    from stylometry import StyleAnalyzer

    sa = StyleAnalyzer()
    sa.fit(known_texts, label="Author A")
    predicted, distances = sa.predict(unknown_text)

:license: MIT
"""

from .analyzer import StyleAnalyzer
from .languages import FUNCTION_WORDS_FR, FUNCTION_WORDS_EN
from .code import CodeAnalyzer, StyleProfile
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stylometry-python")
except PackageNotFoundError:
    # Source tree fallback when distribution metadata is unavailable.
    __version__ = "0.0.0"
__author__ = "Riad Maouchi"
__all__ = [
    "StyleProfile",
    "StyleAnalyzer",
    "CodeAnalyzer",
    "FUNCTION_WORDS_FR",
    "FUNCTION_WORDS_EN",
]
