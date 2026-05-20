"""Functional analyzer workflow tests focused on end-user behavior."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from stylometry import StyleAnalyzer

TEXT_A = (
    "the and of to in the and of to in "
    "the and of to in the and of to in "
    "the and of to in the and of to in"
)
TEXT_B = (
    "in to of and the in to of and the "
    "in to of and the in to of and the "
    "in to of and the in to of and the"
)
TEXT_C = (
    "and the in of to and the in of to "
    "and the in of to and the in of to "
    "and the in of to and the in of to"
)


class TestAnalyzerFunctionalWorkflows:
    def test_load_corpus_reads_txt_files_in_sorted_order(self, tmp_path: Path):
        (tmp_path / "b.txt").write_text("second", encoding="utf-8")
        (tmp_path / "a.txt").write_text("first", encoding="utf-8")

        corpus = StyleAnalyzer.load_corpus(tmp_path)

        assert corpus == ["first", "second"]

    def test_load_corpus_raises_for_empty_directory(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No .txt files"):
            StyleAnalyzer.load_corpus(tmp_path)

    def test_confidence_returns_medium_for_clear_but_not_huge_gap(self):
        analyzer = StyleAnalyzer(min_words=5)

        confidence = analyzer.confidence({"AuthorA": 0.4, "AuthorB": 0.7})

        assert confidence == "MEDIUM"

    def test_confidence_returns_na_when_only_one_label(self):
        analyzer = StyleAnalyzer(min_words=5)

        confidence = analyzer.confidence({"OnlyAuthor": 0.2})

        assert confidence == "N/A"

    def test_plot_fingerprint_returns_figure(self):
        analyzer = StyleAnalyzer(language="en", min_words=5)

        fig = analyzer.plot_fingerprint(
            {"GroupA": [TEXT_A, TEXT_B], "GroupB": [TEXT_C, TEXT_B]},
            top_n=5,
            title="Functional Fingerprint",
        )

        assert fig is not None
        assert fig.axes
        plt.close(fig)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
    def test_plot_clusters_returns_figure(self):
        analyzer = StyleAnalyzer(language="en", min_words=5)

        fig = analyzer.plot_clusters(
            texts_groups=[[TEXT_A, TEXT_B], [TEXT_C, TEXT_B]],
            labels=["GroupA", "GroupB"],
            title="Functional Clusters",
        )

        assert fig is not None
        assert fig.axes
        plt.close(fig)
