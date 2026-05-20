"""
code.py
~~~~~~~

CodeAnalyzer — authorship attribution for source code.

Measures stylistic features that reflect developer habits:
camelCase vs snake_case, comment density, type hints, docstrings, etc.

Reference:
    Caliskan et al. (2015). De-anonymizing Programmers via Code Stylometry.
    USENIX Security Symposium. (95% accuracy on 1,600 GitHub developers)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Feature names — ordered, used as vector dimensions
# ---------------------------------------------------------------------------
CODE_FEATURES = [
    "camelCase_ratio",  # identifiers in camelCase / total identifiers
    "snake_case_ratio",  # identifiers in snake_case / total identifiers
    "comment_density",  # comment lines / non-empty lines
    "docstring_density",  # triple-quote occurrences / non-empty lines
    "type_hint_usage",  # type annotations per non-empty line
    "list_comp_usage",  # list comprehensions per non-empty line
    "avg_line_length",  # mean line length, normalized to [0,1]
    "blank_line_ratio",  # blank lines / total lines
]

# Patterns for Copilot/LLM-generated code
_COPILOT_SIGNALS = {
    "type_hint_usage": 0.10,  # LLMs consistently annotate
    "docstring_density": 0.08,  # LLMs systematically add docstrings
    "camelCase_ratio": 0.05,  # LLMs prefer snake_case in Python
}


class CodeAnalyzer:
    """
    Authorship attribution for source code via stylometric features.

    Parameters
    ----------
    min_lines : int
        Minimum non-empty lines per file for reliable analysis.

    Examples
    --------
    >>> ca = CodeAnalyzer()
    >>> ca.fit(alice_files, "Alice")
    >>> ca.fit(bob_files, "Bob")
    >>> predicted, distances = ca.predict(unknown_file)
    >>> print(ca.copilot_score(unknown_file))
    0.73
    """

    def __init__(self, min_lines: int = 4) -> None:
        self.min_lines = min_lines
        self._centroids: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, code: str) -> dict[str, float]:
        """
        Extract stylometric features from a Python source code string.

        Parameters
        ----------
        code : str — Python source code

        Returns
        -------
        dict[str, float] — one value per feature in CODE_FEATURES

        Raises
        ------
        ValueError if code has fewer lines than min_lines
        """
        lines = code.split("\n")
        non_empty = [line for line in lines if line.strip()]
        n = len(non_empty)

        if n < self.min_lines:
            raise ValueError(
                f"File too short ({n} non-empty lines). " f"Minimum: {self.min_lines}."
            )

        # Identifier extraction
        names = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]+)\b", code)
        n_names = max(len(names), 1)

        camel = [x for x in names if re.search(r"[a-z][A-Z]", x)]
        snake = [x for x in names if "_" in x and x == x.lower() and len(x) > 2]

        return {
            "camelCase_ratio": len(camel) / n_names,
            "snake_case_ratio": len(snake) / n_names,
            "comment_density": sum(1 for line in lines if line.strip().startswith("#"))
            / n,
            "docstring_density": len(re.findall(r'"""', code)) / n,
            "type_hint_usage": len(
                re.findall(
                    r":\s*(int|str|float|bool|list|dict|tuple|Optional|Union|Any)", code
                )
            )
            / n,
            "list_comp_usage": len(re.findall(r"\[.+for .+in .+\]", code)) / n,
            "avg_line_length": min(
                np.mean([len(line) for line in non_empty]) / 100, 1.0
            ),
            "blank_line_ratio": sum(1 for line in lines if not line.strip())
            / max(len(lines), 1),
        }

    def vectorize(self, code: str) -> np.ndarray:
        """
        Convert source code to a L2-normalized style vector.

        Parameters
        ----------
        code : str

        Returns
        -------
        np.ndarray of shape (len(CODE_FEATURES),)
        """
        f = self.extract_features(code)
        v = np.array([f[k] for k in CODE_FEATURES], dtype=np.float64)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def fit(self, code_files: Sequence[str], label: str) -> "CodeAnalyzer":
        """
        Register a centroid from a list of source code strings.

        Parameters
        ----------
        code_files : list of str — source code samples for this developer
        label      : str — developer name or identifier

        Returns
        -------
        self (chainable)
        """
        vecs = np.array([self.vectorize(c) for c in code_files])
        self._centroids[label] = vecs.mean(axis=0)
        return self

    def predict(self, code: str) -> tuple[str, dict[str, float]]:
        """
        Attribute a code file to the nearest registered centroid.

        Parameters
        ----------
        code : str — source code to attribute

        Returns
        -------
        (predicted_label, {label: cosine_distance, ...})
        """
        if not self._centroids:
            raise RuntimeError(
                "No centroids registered. Call fit(code_files, label) first."
            )
        v = self.vectorize(code)
        distances = {
            label: float(1.0 - np.dot(v, centroid))
            for label, centroid in self._centroids.items()
        }
        return min(distances, key=distances.get), distances

    def copilot_score(self, code: str) -> float:
        """
        Estimate the likelihood that code was generated or heavily assisted
        by Copilot / an LLM coder.

        Based on the presence of characteristic LLM patterns:
        - systematic type hints
        - systematic docstrings
        - snake_case naming (Python-idiomatic)
        - verbose parameter names

        Returns
        -------
        float in [0, 1] — higher = more likely LLM-generated
        """
        try:
            f = self.extract_features(code)
        except ValueError:
            return 0.0

        score = 0.0
        total_weight = 0.0

        for feature, threshold in _COPILOT_SIGNALS.items():
            weight = 1.0
            total_weight += weight
            if f[feature] >= threshold:
                score += weight * min(f[feature] / (threshold * 2), 1.0)

        # Bonus: very long average identifier names (LLMs are verbose)
        names = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]+)\b", code)
        if names:
            avg_len = np.mean([len(n) for n in names if len(n) > 1])
            if avg_len > 12:
                score += 0.5
                total_weight += 0.5

        return min(score / max(total_weight, 1.0), 1.0)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def load_file(path: str | Path) -> str:
        """Load a source code file."""
        return Path(path).read_text(encoding="utf-8")

    @staticmethod
    def load_directory(path: str | Path, extension: str = "*.py") -> list[str]:
        """
        Load all files matching an extension from a directory.

        Parameters
        ----------
        path      : str or Path
        extension : glob pattern, e.g. '*.py', '*.java', '*.ts'
        """
        files = [
            f.read_text(encoding="utf-8") for f in sorted(Path(path).glob(extension))
        ]
        if not files:
            raise FileNotFoundError(f"No {extension} files found in {path}")
        return files

    def feature_table(self, codes: dict[str, list[str]]) -> str:
        """
        Return a formatted comparison table of features across groups.

        Parameters
        ----------
        codes : {"Dev A": [code, ...], "Dev B": [...]}

        Returns
        -------
        str — formatted ASCII table
        """
        group_means: dict[str, dict[str, float]] = {}
        for label, files in codes.items():
            all_f = [self.extract_features(c) for c in files]
            group_means[label] = {
                k: np.mean([f[k] for f in all_f]) for k in CODE_FEATURES
            }

        labels = list(group_means.keys())
        header = f"  {'Feature':25s}" + "".join(
            f"  {label_name:>12s}" for label_name in labels
        )
        sep = "─" * len(header)
        rows = [header, sep]

        for feat in CODE_FEATURES:
            vals = [group_means[label_name][feat] for label_name in labels]
            max_v = max(vals)
            row = f"  {feat:25s}"
            for v in vals:
                marker = " ←" if v == max_v and max_v > 0.005 else "  "
                row += f"  {v:>10.3f}{marker}"
            rows.append(row)

        return "\n".join(rows)

    def __repr__(self) -> str:
        return (
            f"CodeAnalyzer("
            f"features={len(CODE_FEATURES)}, "
            f"centroids={list(self._centroids.keys())})"
        )


# Re-export StyleProfile at module level for convenience
# (defined in code.py directly below)


class StyleProfile:
    """
    Stylometric profile for a developer — centroid of their code samples.

    Parameters
    ----------
    label     : str — developer identifier (name, email, etc.)
    min_lines : int — minimum lines per file

    Examples
    --------
    >>> profile = StyleProfile("alice@dev.io")
    >>> profile.add_files(["def foo(): pass\\n" * 10])
    >>> print(profile.features)
    """

    def __init__(self, label: str = "", min_lines: int = 3) -> None:
        self.label = label
        self.min_lines = min_lines
        self._vectors: list[np.ndarray] = []
        self._features: list[dict] = []

    def add(self, source: str) -> "StyleProfile":
        """Add a source code sample to this profile."""
        try:
            ca = CodeAnalyzer(min_lines=self.min_lines)
            f = ca.extract_features(source)
            v = ca.vectorize(source)
            self._features.append(f)
            self._vectors.append(v)
        except ValueError:
            pass
        return self

    def add_files(self, sources) -> "StyleProfile":
        """Add multiple source code samples."""
        for s in sources:
            self.add(s)
        return self

    @property
    def centroid(self):
        if not self._vectors:
            return None
        return np.mean(self._vectors, axis=0)

    @property
    def features(self):
        if not self._features:
            return None
        return {
            k: float(np.mean([f[k] for f in self._features])) for k in CODE_FEATURES
        }

    @property
    def copilot_score(self) -> float:
        if not self._features:
            return 0.0
        f = self.features
        score = 0.0
        weight_total = 0.0
        for feat, threshold in _COPILOT_SIGNALS.items():
            w = 1.0
            weight_total += w
            if f[feat] >= threshold:
                score += w * min(f[feat] / (threshold * 2), 1.0)
        return min(score / max(weight_total, 1.0), 1.0)

    @property
    def n_files(self) -> int:
        return len(self._vectors)

    def __repr__(self) -> str:
        return f"StyleProfile(label={self.label!r}, n_files={self.n_files})"


# ---------------------------------------------------------------------------
# Module-level convenience functions (delegate to CodeAnalyzer)
# ---------------------------------------------------------------------------

_default_analyzer = CodeAnalyzer()


def extract_features(source: str, min_lines: int = 5) -> dict:
    """Module-level convenience: extract code features. See CodeAnalyzer."""
    ca = CodeAnalyzer(min_lines=min_lines)
    return ca.extract_features(source)


def vectorize(source: str, min_lines: int = 5) -> "np.ndarray":
    """Module-level convenience: vectorize code. See CodeAnalyzer."""
    ca = CodeAnalyzer(min_lines=min_lines)
    return ca.vectorize(source)


def copilot_score(source: str) -> float:
    """Module-level convenience: copilot score. See CodeAnalyzer."""
    return _default_analyzer.copilot_score(source)
