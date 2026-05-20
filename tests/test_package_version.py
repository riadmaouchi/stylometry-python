"""Tests for package version fallback behavior."""

import importlib
import importlib.metadata as metadata

import stylometry


class TestPackageVersion:
    def test_package_version_falls_back_when_distribution_metadata_missing(
        self, monkeypatch
    ):
        def raise_package_not_found(_: str) -> str:
            raise metadata.PackageNotFoundError()

        monkeypatch.setattr(metadata, "version", raise_package_not_found)

        reloaded = importlib.reload(stylometry)
        assert reloaded.__version__ == "0.0.0"

        importlib.reload(stylometry)
