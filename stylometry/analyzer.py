"""
analyzer.py
~~~~~~~~~~~

Core StyleAnalyzer — text authorship attribution via function word analysis.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from .languages import get_function_words

# ---------------------------------------------------------------------------
# Internal plot helpers — dark theme matching Devoxx aesthetic
# ---------------------------------------------------------------------------
_PALETTE = ["#E63946", "#2EC4B6", "#457B9D", "#F4A261", "#A8DADC", "#E9C46A"]
_BG = "#0D1117"
_AX = "#161B22"
_MUTED = "#8B949E"
_WHITE = "#C9D1D9"


class StyleAnalyzer:
    """
    Authorship fingerprinting via function word frequency analysis.

    Parameters
    ----------
    function_words : list[str] | None
        Custom vocabulary. If None, uses the language preset.
    language : str
        Language code ('fr', 'en'). Used only when function_words is None.
    min_words : int
        Minimum token count per text. Shorter texts produce unreliable vectors.

    Examples
    --------
    >>> sa = StyleAnalyzer()
    >>> sa.fit(zola_corpus, "Zola").fit(maupassant_corpus, "Maupassant")
    >>> predicted, distances = sa.predict(unknown_text)
    >>> print(predicted)
    'Zola'
    """

    def __init__(
        self,
        function_words: list[str] | None = None,
        language: str = "fr",
        min_words: int = 50,
    ) -> None:
        if function_words is not None:
            self.function_words = function_words
        else:
            self.function_words = get_function_words(language)

        self.min_words = min_words
        self._centroids: dict[str, np.ndarray] = {}
        self._corpora: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Vectorization
    # ------------------------------------------------------------------

    def vectorize(self, text: str) -> np.ndarray:
        """
        Convert a text to a L2-normalized style vector.

        Parameters
        ----------
        text : str — raw text (any case, any punctuation)

        Returns
        -------
        np.ndarray of shape (len(function_words),)

        Raises
        ------
        ValueError if text is too short (< min_words tokens)
        """
        tokens = re.findall(r"\b\w+\b", text.lower())
        n = len(tokens)
        if n < self.min_words:
            raise ValueError(
                f"Text too short ({n} words). "
                f"Minimum: {self.min_words}. "
                "Stylometric signals are unreliable on short texts."
            )
        counts = Counter(tokens)
        vec = np.array(
            [counts.get(w, 0) / n for w in self.function_words], dtype=np.float64
        )
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def vectorize_batch(self, texts: Sequence[str]) -> np.ndarray:
        """
        Vectorize multiple texts.

        Parameters
        ----------
        texts : sequence of str

        Returns
        -------
        np.ndarray of shape (len(texts), len(function_words))
        """
        return np.array([self.vectorize(t) for t in texts])

    # ------------------------------------------------------------------
    # Distance metrics
    # ------------------------------------------------------------------

    def cosine_distance(self, text_a: str, text_b: str) -> float:
        """
        Cosine distance between two texts in style space.

        Returns a value in [0, 1]:
        - 0.0 = identical style vectors
        - 1.0 = maximally different

        Parameters
        ----------
        text_a, text_b : str
        """
        return float(1.0 - np.dot(self.vectorize(text_a), self.vectorize(text_b)))

    def shift(self, original: str, rewrite: str) -> float:
        """
        Stylistic shift introduced by a rewrite.

        Measures how much the style vector moved from original to rewrite.
        A shift near 0 means the rewrite preserved the writing style.
        A shift ~0.24 is typical for GPT-4 rewrites.

        Parameters
        ----------
        original : str — source text
        rewrite  : str — rewritten version

        Returns
        -------
        float in [0, 1]
        """
        return self.cosine_distance(original, rewrite)

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def fit(self, texts: Sequence[str], label: str) -> "StyleAnalyzer":
        """
        Register a centroid for a group of texts.

        Parameters
        ----------
        texts : list of str — representative texts for this author/class
        label : str — name for this author/class

        Returns
        -------
        self (chainable)

        Example
        -------
        >>> sa.fit(zola_corpus, "Zola").fit(maupassant_corpus, "Maupassant")
        """
        vectors = self.vectorize_batch(texts)
        self._centroids[label] = vectors.mean(axis=0)
        self._corpora[label] = list(texts)
        return self

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """
        Attribute a text to the nearest registered centroid.

        Parameters
        ----------
        text : str

        Returns
        -------
        (predicted_label, {label: cosine_distance, ...})

        Raises
        ------
        RuntimeError if no centroids have been registered (call fit() first)

        Example
        -------
        >>> predicted, distances = sa.predict(unknown_text)
        >>> print(f"{predicted} ({sa.confidence(distances)} confidence)")
        'Zola (HIGH confidence)'
        """
        if not self._centroids:
            raise RuntimeError("No centroids registered. Call fit(texts, label) first.")
        v = self.vectorize(text)
        distances = {
            label: float(1.0 - np.dot(v, centroid))
            for label, centroid in self._centroids.items()
        }
        predicted = min(distances, key=distances.get)
        return predicted, distances

    def confidence(self, distances: dict[str, float]) -> str:
        """
        Human-readable confidence level for a prediction.

        Based on the relative gap between best and second-best distance.

        Returns
        -------
        "HIGH" | "MEDIUM" | "LOW"
        """
        if len(distances) < 2:
            return "N/A"
        sorted_d = sorted(distances.values())
        gap = sorted_d[1] - sorted_d[0]
        ratio = gap / max(sorted_d[0], 1e-9)
        if ratio > 1.5:
            return "HIGH"
        if ratio > 0.3:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Corpus utilities
    # ------------------------------------------------------------------

    @staticmethod
    def load_corpus(path: str | Path) -> list[str]:
        """
        Load all .txt files from a directory.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        list of str (one per file, sorted by filename)
        """
        p = Path(path)
        texts = [f.read_text(encoding="utf-8").strip() for f in sorted(p.glob("*.txt"))]
        if not texts:
            raise FileNotFoundError(f"No .txt files found in {p}")
        return texts

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_fingerprint(
        self,
        texts_dict: dict[str, list[str]],
        top_n: int = 15,
        title: str = "Style fingerprints",
        figsize: tuple[int, int] = (14, 6),
    ):
        """
        Bar chart comparing function word frequencies across groups.

        Parameters
        ----------
        texts_dict : {"Group A": [text, ...], "Group B": [...]}
        top_n      : number of most discriminating words to show
        title      : chart title
        figsize    : (width, height) in inches

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required: pip install matplotlib")

        group_means: dict[str, np.ndarray] = {
            label: self.vectorize_batch(texts).mean(axis=0)
            for label, texts in texts_dict.items()
        }

        all_means = np.array(list(group_means.values()))
        top_idx = np.argsort(np.std(all_means, axis=0))[-top_n:][::-1]
        top_words = [self.function_words[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_AX)

        n_groups = len(group_means)
        w = 0.8 / n_groups
        x = np.arange(top_n)

        for k, (label, means) in enumerate(group_means.items()):
            offset = (k - n_groups / 2 + 0.5) * w
            vals = [means[i] for i in top_idx]
            color = _PALETTE[k % len(_PALETTE)]
            bars = ax.bar(
                x + offset,
                vals,
                w,
                label=label,
                color=color,
                edgecolor=_BG,
                linewidth=0.4,
            )
            for bar, v in zip(bars, vals):
                if v > 0.02:
                    ax.text(
                        bar.get_x() + w / 2,
                        v + 0.001,
                        f"{v:.2f}",
                        ha="center",
                        fontsize=8,
                        color=color,
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(top_words, fontsize=11, color=_WHITE, fontweight="bold")
        ax.set_ylabel("Relative frequency", fontsize=11, color=_MUTED)
        ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=12)
        ax.tick_params(colors=_MUTED)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color("#374151")
        ax.grid(axis="y", color="#1F2937", linewidth=0.8)
        leg = ax.legend(
            fontsize=11, framealpha=0.15, labelcolor="white", edgecolor="#374151"
        )
        leg.get_frame().set_facecolor("#1C2128")
        plt.tight_layout()
        return fig

    def plot_clusters(
        self,
        texts_groups: list[list[str]],
        labels: list[str],
        title: str = "Stylistic space — PCA projection",
        figsize: tuple[int, int] = (12, 8),
    ):
        """
        PCA scatter plot — texts colored by group.

        Parameters
        ----------
        texts_groups : list of text lists, one per group
        labels       : group names (same order as texts_groups)
        title        : chart title
        figsize      : (width, height) in inches

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                "matplotlib and scikit-learn required: "
                "pip install matplotlib scikit-learn"
            )

        all_vecs, all_labels = [], []
        for texts, label in zip(texts_groups, labels):
            for t in texts:
                all_vecs.append(self.vectorize(t))
                all_labels.append(label)

        coords = PCA(n_components=2, random_state=42).fit_transform(np.array(all_vecs))

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_AX)

        for k, label in enumerate(labels):
            mask = np.array([group_label == label for group_label in all_labels])
            pts = coords[mask]
            color = _PALETTE[k % len(_PALETTE)]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=220,
                c=color,
                alpha=0.75,
                edgecolors="black",
                linewidth=0.8,
                label=label,
                zorder=3,
            )
            cx, cy = pts.mean(axis=0)
            ax.scatter(
                cx,
                cy,
                s=500,
                c=color,
                marker="X",
                edgecolors="black",
                linewidth=2.5,
                zorder=5,
            )
            ax.annotate(
                f"  {label}", (cx, cy), fontsize=11, fontweight="bold", color=color
            )

        ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=15)
        ax.tick_params(colors=_MUTED)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color("#374151")
        ax.grid(alpha=0.12, linestyle="--", color="#374151")
        leg = ax.legend(
            fontsize=11, framealpha=0.15, labelcolor="white", edgecolor="#374151"
        )
        leg.get_frame().set_facecolor("#1C2128")
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (
            f"StyleAnalyzer("
            f"language={len(self.function_words)} words, "
            f"centroids={list(self._centroids.keys())}, "
            f"min_words={self.min_words})"
        )
