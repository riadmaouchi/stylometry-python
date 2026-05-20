"""
Tests for StyleAnalyzer.
Run: pytest tests/ -v
"""

import numpy as np
import pytest
from stylometry import StyleAnalyzer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ZOLA = """
Quand Etienne Lantier descendit dans la fosse, il faisait nuit encore.
La pluie tombait par rafales sur la plaine rase, un vent du nord-est soufflait,
glacial. L homme s arreta au bord du puits, la lampe a la main, et regarda l abime.
Au fond, dans les tenebres, on entendait gronder la machine et les cables fuyaient.
Il hesita, pris d un malaise. Mais le bruit des sabots derriere lui le decida.
Toute une foule se pressait, des hommes, des femmes, des enfants.
"""

MAUPASSANT = """
Il pleuvait a verse. La nuit etait noire. Sur la route deserte,
un homme marchait, courbe sous l averse. Il allait vite, tres vite,
comme s il fuyait quelque chose. De temps en temps, il jetait un regard derriere lui.
Il pensait a elle. Il pensait a ce qu il avait fait. Il marchait toujours,
sans savoir ou il allait, sans savoir quand il s arreterait.
La pluie ruisselait sur son visage, et il ne sentait plus rien.
"""

GPT_REWRITE = """
Etienne Lantier descendait dans la mine lorsqu il faisait encore nuit.
La pluie tombait de maniere intense sur la plaine, accompagnee d un vent froid.
Il s arreta au bord du puits avec sa lampe et observa l obscurite en dessous.
On pouvait entendre le bruit de la machine. Il hesita un moment avant de continuer.
Une foule de personnes attendait derriere lui pour descendre a leur tour dans la mine.
"""


# ---------------------------------------------------------------------------
# Tests: vectorize
# ---------------------------------------------------------------------------


class TestVectorize:

    def test_returns_numpy_array(self):
        sa = StyleAnalyzer()
        v = sa.vectorize(ZOLA)
        assert isinstance(v, np.ndarray)

    def test_correct_dimension(self):
        sa = StyleAnalyzer()
        v = sa.vectorize(ZOLA)
        assert v.shape == (len(sa.function_words),)

    def test_l2_normalized(self):
        sa = StyleAnalyzer()
        v = sa.vectorize(ZOLA)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_too_short_raises(self):
        sa = StyleAnalyzer(min_words=50)
        with pytest.raises(ValueError, match="too short"):
            sa.vectorize("Bonjour.")

    def test_deterministic(self):
        sa = StyleAnalyzer()
        assert np.allclose(sa.vectorize(ZOLA), sa.vectorize(ZOLA))


# ---------------------------------------------------------------------------
# Tests: cosine_distance
# ---------------------------------------------------------------------------


class TestCosineDistance:

    def test_same_text_zero(self):
        sa = StyleAnalyzer()
        d = sa.cosine_distance(ZOLA, ZOLA)
        assert d < 1e-6

    def test_different_texts_positive(self):
        sa = StyleAnalyzer()
        d = sa.cosine_distance(ZOLA, MAUPASSANT)
        assert d > 0

    def test_distance_in_range(self):
        sa = StyleAnalyzer()
        d = sa.cosine_distance(ZOLA, MAUPASSANT)
        assert 0.0 <= d <= 1.0

    def test_symmetric(self):
        sa = StyleAnalyzer()
        assert (
            abs(
                sa.cosine_distance(ZOLA, MAUPASSANT)
                - sa.cosine_distance(MAUPASSANT, ZOLA)
            )
            < 1e-10
        )


# ---------------------------------------------------------------------------
# Tests: shift
# ---------------------------------------------------------------------------


class TestShift:

    def test_shift_nonzero_for_rewrite(self):
        sa = StyleAnalyzer()
        s = sa.shift(ZOLA, GPT_REWRITE)
        assert s > 0.01, "GPT rewrite should shift the style vector"

    def test_shift_in_range(self):
        sa = StyleAnalyzer()
        s = sa.shift(ZOLA, GPT_REWRITE)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Tests: fit + predict
# ---------------------------------------------------------------------------


class TestAttribution:

    def test_predict_requires_fit(self):
        sa = StyleAnalyzer()
        with pytest.raises(RuntimeError, match="No centroids"):
            sa.predict(ZOLA)

    def test_fit_is_chainable(self):
        sa = StyleAnalyzer()
        result = sa.fit([ZOLA], "Zola").fit([MAUPASSANT], "Maupassant")
        assert result is sa

    def test_predict_returns_tuple(self):
        sa = StyleAnalyzer()
        sa.fit([ZOLA] * 3, "Zola")
        sa.fit([MAUPASSANT] * 3, "Maupassant")
        predicted, distances = sa.predict(ZOLA)
        assert isinstance(predicted, str)
        assert isinstance(distances, dict)

    def test_predict_correct_author(self):
        sa = StyleAnalyzer()
        sa.fit([ZOLA] * 3, "Zola")
        sa.fit([MAUPASSANT] * 3, "Maupassant")
        predicted, _ = sa.predict(ZOLA)
        assert predicted == "Zola"

    def test_distances_contain_all_labels(self):
        sa = StyleAnalyzer()
        sa.fit([ZOLA], "Zola")
        sa.fit([MAUPASSANT], "Maupassant")
        _, distances = sa.predict(ZOLA)
        assert set(distances.keys()) == {"Zola", "Maupassant"}


# ---------------------------------------------------------------------------
# Tests: confidence
# ---------------------------------------------------------------------------


class TestConfidence:

    def test_returns_string(self):
        sa = StyleAnalyzer()
        c = sa.confidence({"A": 0.1, "B": 0.9})
        assert isinstance(c, str)

    def test_clear_winner_is_high(self):
        sa = StyleAnalyzer()
        c = sa.confidence({"A": 0.05, "B": 0.95})
        assert c == "HIGH"

    def test_close_scores_are_low(self):
        sa = StyleAnalyzer()
        c = sa.confidence({"A": 0.48, "B": 0.52})
        assert c == "LOW"
