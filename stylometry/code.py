"""
code.py
~~~~~~~

CodeAnalyzer — authorship attribution and LLM detection for source code.

Supports multiple languages: Python, JavaScript, TypeScript, C, Ruby, Go, Rust.

Reference:
    Caliskan et al. (2015). De-anonymizing Programmers via Code Stylometry.
    USENIX Security Symposium. (95% accuracy on 1,600 GitHub developers)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = frozenset({
    "python", "javascript", "typescript", "c", "ruby", "go", "rust",
})

# Extension → language mapping (used by auto-detection)
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".c": "c", ".h": "c", ".cpp": "c", ".cc": "c",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
}

# ---------------------------------------------------------------------------
# Language-specific patterns
# ---------------------------------------------------------------------------

_COMMENT_PREFIXES: dict[str, list[str]] = {
    "python":     [r"^\s*#"],
    "javascript": [r"^\s*//", r"^\s*/\*", r"^\s*\*"],
    "typescript": [r"^\s*//", r"^\s*/\*", r"^\s*\*"],
    "c":          [r"^\s*//", r"^\s*/\*", r"^\s*\*"],
    "ruby":       [r"^\s*#"],
    "go":         [r"^\s*//"],
    "rust":       [r"^\s*///", r"^\s*//"],
}

_DOCSTRING_RE: dict[str, re.Pattern[str]] = {
    # Python uses a line-by-line approach (_count_py_documented_fns) — no pattern here
    "python":     re.compile(r'(?!)'),  # never matches; handled separately
    "javascript": re.compile(r'/\*\*.*?\*/\s*\n?\s*(?:(?:async\s+)?function\s+\w+|\w+\s*[=:]\s*(?:async\s*)?\()', re.DOTALL),
    "typescript": re.compile(r'/\*\*.*?\*/\s*\n?\s*(?:(?:async\s+)?function\s+\w+|\w+\s*[=:]\s*(?:async\s*)?\()', re.DOTALL),
    "go":         re.compile(r'//\s*\w+[^\n]*\n\s*func\s+'),
    "rust":       re.compile(r'///[^\n]*\n\s*(?:pub\s+)?(?:async\s+)?fn\s+'),
    "c":          re.compile(r'/\*\*.*?\*/\s*\n?\s*\w+\s+\w+\s*\(', re.DOTALL),
    "ruby":       re.compile(r'#\s*[A-Z][^\n]+\n\s*def\s+\w+'),
}

_FUNCTION_RE: dict[str, re.Pattern[str]] = {
    "python":     re.compile(r'^\s*(?:async\s+)?def\s+\w+', re.MULTILINE),
    "javascript": re.compile(r'(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\()', re.MULTILINE),
    "typescript": re.compile(r'(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(|(?:public|private|protected|async)\s+\w+\s*\()', re.MULTILINE),
    "c":          re.compile(r'^\w[\w\s\*]+\w\s*\([^;{]*\)\s*\{', re.MULTILINE),
    "ruby":       re.compile(r'^\s*def\s+\w+', re.MULTILINE),
    "go":         re.compile(r'^func\s+', re.MULTILINE),
    "rust":       re.compile(r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+\w+', re.MULTILINE),
}

_ERROR_HANDLING_RE: dict[str, list[re.Pattern[str]]] = {
    "python": [
        re.compile(r'^\s*try\s*:'),
        re.compile(r'^\s*except\b'),
        re.compile(r'^\s*raise\b'),
    ],
    "javascript": [
        re.compile(r'^\s*try\s*\{'),
        re.compile(r'catch\s*\('),
        re.compile(r'\bthrow\s+(?:new\s+)?\w+'),
    ],
    "typescript": [
        re.compile(r'^\s*try\s*\{'),
        re.compile(r'catch\s*\('),
        re.compile(r'\bthrow\s+(?:new\s+)?\w+'),
    ],
    "c": [
        re.compile(r'if\s*\(\s*(?:err|error|ret|result|status)\s*[<>!=]'),
        re.compile(r'return\s+-1\s*;'),
        re.compile(r'goto\s+\w+\s*;'),
    ],
    "ruby": [
        re.compile(r'^\s*begin\s*$'),
        re.compile(r'^\s*rescue\b'),
        re.compile(r'^\s*raise\b'),
    ],
    "go": [
        re.compile(r'if\s+\w*err\w*\s*!=\s*nil'),
        re.compile(r'return\s+.*\berr\b'),
    ],
    "rust": [
        re.compile(r'\?\s*$'),
        re.compile(r'\.unwrap_or'),
        re.compile(r'\bunwrap\(\)'),
        re.compile(r'match\s+\w+\s*\{'),
    ],
}

_FUNCTION_WORD_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]+)\b')
_DEF_LINE_RE = re.compile(r'^\s*(?:async\s+)?def\s+\w+')
_DOCSTRING_OPEN_RE = re.compile(r'^\s*("""|\'\'\').')


def _count_py_documented_fns(code: str) -> tuple[int, int]:
    """Return (n_documented, n_total) for Python, handling multi-line signatures."""
    lines = code.split('\n')
    n_total, n_documented = 0, 0
    i = 0
    while i < len(lines):
        if _DEF_LINE_RE.match(lines[i]):
            n_total += 1
            # Walk forward to find the ':' that closes the signature
            j = i
            found_colon = False
            while j < min(i + 20, len(lines)):
                if lines[j].rstrip().endswith(':'):
                    found_colon = True
                    break
                j += 1
            if found_colon:
                # Next non-empty line after the signature
                k = j + 1
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if k < len(lines):
                    stripped = lines[k].strip()
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        n_documented += 1
        i += 1
    return n_documented, n_total


_TYPE_HINT_RE: dict[str, re.Pattern[str] | None] = {
    "python":     re.compile(r':\s*(?:int|str|float|bool|list|dict|tuple|Optional|Union|Any|Sequence)'),
    "typescript": re.compile(r':\s*(?:string|number|boolean|void|any|never|unknown|object)\b'),
    "javascript": None,
    "c":          None,
    "ruby":       None,
    "go":         None,
    "rust":       re.compile(r'->\s*(?:\w+|&\w+|Option<|Result<)'),
}

# ---------------------------------------------------------------------------
# Feature names — ordered vector dimensions
# ---------------------------------------------------------------------------

CODE_FEATURES = [
    # Naming conventions
    "camelCase_ratio",        # identifiers in camelCase / total identifiers
    "snake_case_ratio",       # identifiers in snake_case / total identifiers
    # Comment & documentation
    "comment_density",        # comment lines / non-empty lines
    "docstring_density",      # docstring occurrences / non-empty lines
    "docstring_completeness", # documented functions / total functions  [NEW 1.4]
    # Language-specific style
    "type_hint_usage",        # type annotations per non-empty line
    "list_comp_usage",        # list comprehensions per non-empty line (Python)
    # LLM-associated signals                               [NEW 1.4]
    "error_handling_density", # try/catch/rescue constructs per 100 lines
    "identifier_verbosity",   # avg identifier length, normalized to [0, 1]
    # Layout
    "avg_line_length",        # mean line length, normalized to [0, 1]
    "blank_line_ratio",       # blank lines / total lines
]

# ---------------------------------------------------------------------------
# LLM signal thresholds for copilot_score()
# Each entry: (feature, weight, organic_baseline, llm_typical)
# ---------------------------------------------------------------------------
_LLM_SIGNALS: list[tuple[str, float, float, float]] = [
    ("comment_density",        0.15,  0.08, 0.22),
    ("docstring_completeness", 0.25,  0.15, 0.65),
    ("identifier_verbosity",   0.20,  0.40, 0.65),   # normalized: 8→0.40, 13→0.65
    ("error_handling_density", 0.15,  0.02, 0.08),
    ("type_hint_usage",        0.15,  0.05, 0.18),
    ("docstring_density",      0.10,  0.05, 0.15),
]


# ---------------------------------------------------------------------------
# Helper: compile comment patterns for a language
# ---------------------------------------------------------------------------

def _compile_comment_patterns(language: str) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in _COMMENT_PREFIXES.get(language, _COMMENT_PREFIXES["python"])]


# ---------------------------------------------------------------------------
# CodeAnalyzer
# ---------------------------------------------------------------------------

class CodeAnalyzer:
    """
    Authorship attribution and LLM detection for source code.

    Supports Python, JavaScript, TypeScript, C, Ruby, Go, and Rust.

    Parameters
    ----------
    language : str
        Source language for feature extraction.
        One of: 'python', 'javascript', 'typescript', 'c', 'ruby', 'go', 'rust'.
        Default 'python' for backward compatibility.
    min_lines : int
        Minimum non-empty lines per snippet for reliable analysis.

    Examples
    --------
    >>> ca = CodeAnalyzer(language="typescript")
    >>> ca.fit(alice_files, "Alice").fit(bob_files, "Bob")
    >>> predicted, distances = ca.predict(unknown_file)
    >>> print(ca.copilot_score(unknown_file))
    0.71
    """

    def __init__(
        self,
        language: str = "python",
        min_lines: int = 4,
    ) -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Supported: {sorted(SUPPORTED_LANGUAGES)}. "
                "Use CodeAnalyzer.from_extension(path) for automatic detection."
            )
        self.language = language
        self.min_lines = min_lines
        self._centroids: dict[str, np.ndarray] = {}
        self._comment_patterns = _compile_comment_patterns(language)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_extension(cls, path: str | Path, min_lines: int = 4) -> "CodeAnalyzer":
        """
        Create a CodeAnalyzer with language inferred from the file extension.

        Falls back to 'python' for unknown extensions.
        """
        suffix = Path(path).suffix.lower()
        language = _EXT_TO_LANG.get(suffix, "python")
        return cls(language=language, min_lines=min_lines)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, code: str) -> dict[str, float]:
        """
        Extract stylometric features from source code.

        Parameters
        ----------
        code : str — source code in the configured language

        Returns
        -------
        dict[str, float] — one value per feature in CODE_FEATURES, all in [0, 1]

        Raises
        ------
        ValueError if code has fewer non-empty lines than min_lines
        """
        lines = code.split("\n")
        non_empty = [line for line in lines if line.strip()]
        n = len(non_empty)

        if n < self.min_lines:
            raise ValueError(
                f"Code too short ({n} non-empty lines). Minimum: {self.min_lines}."
            )

        # --- Identifiers ---
        names = _FUNCTION_WORD_RE.findall(code)
        n_names = max(len(names), 1)
        camel = [x for x in names if re.search(r"[a-z][A-Z]", x)]
        snake = [x for x in names if "_" in x and x == x.lower() and len(x) > 2]
        avg_id_len = float(np.mean([len(x) for x in names])) if names else 0.0

        # --- Comments ---
        n_comments = sum(
            1 for line in non_empty
            if any(p.match(line) for p in self._comment_patterns)
        )

        # --- Docstrings & function coverage ---
        if self.language == "python":
            n_documented, n_functions = _count_py_documented_fns(code)
            n_docstrings_raw = n_documented
            doc_completeness = n_documented / n_functions if n_functions > 0 else 0.0
        else:
            doc_re = _DOCSTRING_RE[self.language]
            fn_re = _FUNCTION_RE.get(self.language)
            n_docstrings_raw = len(doc_re.findall(code))
            n_functions = len(fn_re.findall(code)) if fn_re else 0
            doc_completeness = n_docstrings_raw / n_functions if n_functions > 0 else 0.0

        # --- Error handling ---
        err_patterns = _ERROR_HANDLING_RE.get(self.language, [])
        n_error = sum(
            1 for line in non_empty
            if any(p.search(line) for p in err_patterns)
        )
        error_density = min(n_error / n * 100 / 15.0, 1.0)  # normalize: 15% = 1.0

        # --- Type hints ---
        type_re = _TYPE_HINT_RE.get(self.language)
        n_type_hints = len(type_re.findall(code)) / n if type_re else 0.0

        # --- List comprehensions (Python only) ---
        list_comp = len(re.findall(r"\[.+for .+in .+\]", code)) / n

        # --- Layout ---
        avg_len = float(np.mean([len(line) for line in non_empty]))

        return {
            "camelCase_ratio":        min(len(camel) / n_names, 1.0),
            "snake_case_ratio":       min(len(snake) / n_names, 1.0),
            "comment_density":        min(n_comments / n, 1.0),
            "docstring_density":      min(n_docstrings_raw / n, 1.0),
            "docstring_completeness": min(doc_completeness, 1.0),
            "type_hint_usage":        min(n_type_hints, 1.0),
            "list_comp_usage":        min(list_comp, 1.0),
            "error_handling_density": error_density,
            "identifier_verbosity":   min(avg_id_len / 20.0, 1.0),
            "avg_line_length":        min(avg_len / 100.0, 1.0),
            "blank_line_ratio":       sum(1 for l in lines if not l.strip()) / max(len(lines), 1),
        }

    def vectorize(self, code: str) -> np.ndarray:
        """
        Convert source code to a L2-normalized style vector.

        Returns
        -------
        np.ndarray of shape (len(CODE_FEATURES),)
        """
        f = self.extract_features(code)
        v = np.array([f[k] for k in CODE_FEATURES], dtype=np.float64)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    # ------------------------------------------------------------------
    # LLM detection
    # ------------------------------------------------------------------

    def copilot_score(self, code: str) -> float:
        """
        Estimate the likelihood that code was generated or heavily assisted
        by an LLM (Copilot, ChatGPT, Claude, etc.).

        Combines six weighted signals calibrated against pre-2022 OSS baselines.

        Returns
        -------
        float in [0, 1] — higher = more likely LLM-assisted
        """
        try:
            f = self.extract_features(code)
        except ValueError:
            return 0.0

        score = 0.0
        total_weight = 0.0
        for feature, weight, organic, llm_typical in _LLM_SIGNALS:
            v = f.get(feature, 0.0)
            if llm_typical <= organic:
                continue
            normalized = max(0.0, min(1.0, (v - organic) / (llm_typical - organic)))
            score += weight * normalized
            total_weight += weight

        return min(score / max(total_weight, 1e-9), 1.0)

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def fit(self, code_files: Sequence[str], label: str) -> "CodeAnalyzer":
        """
        Register a style centroid from a list of source code strings.

        Parameters
        ----------
        code_files : list of str — source code samples for this developer/group
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
        Attribute source code to the nearest registered centroid.

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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def detect_language(path: str | Path) -> str:
        """Detect language from file extension. Returns 'python' as fallback."""
        return _EXT_TO_LANG.get(Path(path).suffix.lower(), "python")

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
        extension : glob pattern, e.g. '*.py', '*.ts', '*.go'
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
        """
        group_means: dict[str, dict[str, float]] = {}
        for label, files in codes.items():
            all_f = [self.extract_features(c) for c in files]
            group_means[label] = {
                k: float(np.mean([f[k] for f in all_f])) for k in CODE_FEATURES
            }

        labels = list(group_means.keys())
        header = f"  {'Feature':28s}" + "".join(
            f"  {lb:>12s}" for lb in labels
        )
        sep = "─" * len(header)
        rows = [header, sep]

        for feat in CODE_FEATURES:
            vals = [group_means[lb][feat] for lb in labels]
            max_v = max(vals)
            row = f"  {feat:28s}"
            for v in vals:
                marker = " ←" if v == max_v and max_v > 0.005 else "  "
                row += f"  {v:>10.3f}{marker}"
            rows.append(row)

        return "\n".join(rows)

    def __repr__(self) -> str:
        return (
            f"CodeAnalyzer("
            f"language={self.language!r}, "
            f"features={len(CODE_FEATURES)}, "
            f"centroids={list(self._centroids.keys())})"
        )


# ---------------------------------------------------------------------------
# StyleProfile — centroid wrapper
# ---------------------------------------------------------------------------

class StyleProfile:
    """
    Stylometric profile for a developer — centroid of their code samples.

    Parameters
    ----------
    label    : str — developer identifier (name, email, etc.)
    language : str — source language (default 'python')
    min_lines: int — minimum lines per file

    Examples
    --------
    >>> profile = StyleProfile("alice@dev.io", language="typescript")
    >>> profile.add_files(sources)
    >>> print(profile.copilot_score)
    0.68
    """

    def __init__(
        self,
        label: str = "",
        language: str = "python",
        min_lines: int = 3,
    ) -> None:
        self.label = label
        self.language = language
        self.min_lines = min_lines
        self._analyzer = CodeAnalyzer(language=language, min_lines=min_lines)
        self._vectors: list[np.ndarray] = []
        self._features: list[dict[str, float]] = []

    def add(self, source: str) -> "StyleProfile":
        """Add a source code sample to this profile."""
        try:
            f = self._analyzer.extract_features(source)
            v = self._analyzer.vectorize(source)
            self._features.append(f)
            self._vectors.append(v)
        except ValueError:
            pass
        return self

    def add_files(self, sources: Sequence[str]) -> "StyleProfile":
        """Add multiple source code samples."""
        for s in sources:
            self.add(s)
        return self

    @property
    def centroid(self) -> np.ndarray | None:
        if not self._vectors:
            return None
        return np.mean(self._vectors, axis=0)

    @property
    def features(self) -> dict[str, float] | None:
        if not self._features:
            return None
        return {
            k: float(np.mean([f[k] for f in self._features])) for k in CODE_FEATURES
        }

    @property
    def copilot_score(self) -> float:
        f = self.features
        if f is None:
            return 0.0
        score = 0.0
        total_weight = 0.0
        for feature, weight, organic, llm_typical in _LLM_SIGNALS:
            v = f.get(feature, 0.0)
            if llm_typical <= organic:
                continue
            normalized = max(0.0, min(1.0, (v - organic) / (llm_typical - organic)))
            score += weight * normalized
            total_weight += weight
        return min(score / max(total_weight, 1e-9), 1.0)

    @property
    def n_files(self) -> int:
        return len(self._vectors)

    def __repr__(self) -> str:
        return f"StyleProfile(label={self.label!r}, language={self.language!r}, n_files={self.n_files})"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def extract_features(source: str, language: str = "python", min_lines: int = 4) -> dict[str, float]:
    """Convenience: extract code features. See CodeAnalyzer."""
    return CodeAnalyzer(language=language, min_lines=min_lines).extract_features(source)


def vectorize(source: str, language: str = "python", min_lines: int = 4) -> np.ndarray:
    """Convenience: vectorize code. See CodeAnalyzer."""
    return CodeAnalyzer(language=language, min_lines=min_lines).vectorize(source)


def copilot_score(source: str, language: str = "python") -> float:
    """Convenience: estimate LLM influence. See CodeAnalyzer."""
    return CodeAnalyzer(language=language).copilot_score(source)
