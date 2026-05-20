"""Functional code workflow tests focused on end-user behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stylometry.code import (
    CODE_FEATURES,
    CodeAnalyzer,
    StyleProfile,
    copilot_score,
    extract_features,
    vectorize,
)

CODE_LONG = """
def normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []
    total = sum(values)
    return [value / total for value in values if total > 0]
"""

CODE_LONG_ALT = """
def normalize_values(values, reverse=False):
    if values is None:
        return []
    output_values = sorted(values, reverse=reverse)
    return output_values
"""


class TestCodeFunctionalWorkflows:
    def test_load_file_and_load_directory_roundtrip(self, tmp_path: Path):
        (tmp_path / "one.py").write_text(CODE_LONG, encoding="utf-8")
        (tmp_path / "two.py").write_text(CODE_LONG_ALT, encoding="utf-8")

        loaded_file = CodeAnalyzer.load_file(tmp_path / "one.py")
        loaded_dir = CodeAnalyzer.load_directory(tmp_path)

        assert "normalize_values" in loaded_file
        assert len(loaded_dir) == 2

    def test_load_directory_raises_when_no_matching_files(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No .* files found"):
            CodeAnalyzer.load_directory(tmp_path, extension="*.ts")

    def test_predict_raises_without_registered_centroids(self):
        analyzer = CodeAnalyzer(min_lines=4)

        with pytest.raises(RuntimeError, match="No centroids registered"):
            analyzer.predict(CODE_LONG)

    def test_feature_table_contains_all_feature_rows(self):
        analyzer = CodeAnalyzer(min_lines=4)

        table = analyzer.feature_table({"TeamA": [CODE_LONG], "TeamB": [CODE_LONG_ALT]})

        assert "Feature" in table
        for feature_name in CODE_FEATURES:
            assert feature_name in table

    def test_style_profile_ignores_too_short_inputs(self):
        profile = StyleProfile(label="dev@example.com", min_lines=10)

        profile.add("def short():\n    return 1\n")

        assert profile.n_files == 0
        assert profile.centroid is None
        assert profile.features is None

    def test_style_profile_accumulates_files_and_computes_score(self):
        profile = StyleProfile(label="dev@example.com", min_lines=4)

        profile.add_files([CODE_LONG, CODE_LONG_ALT])

        assert profile.n_files == 2
        assert profile.centroid is not None
        assert 0.0 <= profile.copilot_score <= 1.0
        assert "StyleProfile" in repr(profile)

    def test_module_level_helpers_return_expected_shapes(self):
        features = extract_features(CODE_LONG, min_lines=4)
        vec = vectorize(CODE_LONG, min_lines=4)
        score = copilot_score(CODE_LONG)

        assert set(features.keys()) == set(CODE_FEATURES)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (len(CODE_FEATURES),)
        assert 0.0 <= score <= 1.0
