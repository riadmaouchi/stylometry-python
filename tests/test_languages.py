"""Tests for language registry behavior."""

import pytest

from stylometry.languages import (
    FUNCTION_WORDS_EN,
    FUNCTION_WORDS_FR,
    LANGUAGES,
    get_function_words,
)


class TestLanguages:
    def test_get_function_words_returns_french_registry_object(self):
        words = get_function_words("fr")

        assert words is FUNCTION_WORDS_FR
        assert words == LANGUAGES["fr"]

    def test_get_function_words_returns_english_registry_object(self):
        words = get_function_words("en")

        assert words is FUNCTION_WORDS_EN
        assert words == LANGUAGES["en"]

    def test_get_function_words_raises_for_unsupported_language(self):
        with pytest.raises(ValueError, match="Language 'es' not supported") as exc:
            get_function_words("es")

        message = str(exc.value)
        assert "Supported:" in message
        assert "fr" in message and "en" in message
