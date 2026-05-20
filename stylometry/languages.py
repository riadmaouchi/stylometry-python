"""
languages.py
~~~~~~~~~~~~

Function word vocabularies for supported languages.

Function words (articles, pronouns, prepositions, conjunctions) are
style-bearing and topic-independent — they carry authorial fingerprints
regardless of what the text is about.
"""

# ---------------------------------------------------------------------------
# French — 41 mots-fonction
# Validated on PAN @ CLEF stylometry benchmarks
# ---------------------------------------------------------------------------
FUNCTION_WORDS_FR = [
    # Articles
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "de",
    "du",
    # Coordinating conjunctions
    "et",
    "ou",
    "mais",
    "donc",
    "or",
    "ni",
    "car",
    # Relative / subordinating
    "que",
    "qui",
    "dont",
    # Prepositions
    "dans",
    "sur",
    "sous",
    "avec",
    "sans",
    "pour",
    "par",
    # Personal pronouns
    "il",
    "elle",
    "ils",
    "elles",
    "je",
    "tu",
    "nous",
    "vous",
    # Negation & adverbs
    "ne",
    "pas",
    "plus",
    "très",
    "bien",
    "tout",
    "tous",
    "rien",
]

# ---------------------------------------------------------------------------
# English — 45 function words
# ---------------------------------------------------------------------------
FUNCTION_WORDS_EN = [
    # Articles & determiners
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    # Prepositions
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "up",
    "about",
    "into",
    "through",
    "during",
    # Pronouns
    "i",
    "he",
    "she",
    "it",
    "we",
    "they",
    "you",
    "me",
    "him",
    "her",
    "us",
    "them",
    # Conjunctions
    "and",
    "but",
    "or",
    "nor",
    "so",
    "yet",
    # Auxiliaries
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    # Adverbs
    "not",
    "very",
    "just",
    "also",
    "even",
]

# Language registry
LANGUAGES = {
    "fr": FUNCTION_WORDS_FR,
    "en": FUNCTION_WORDS_EN,
}


def get_function_words(language: str) -> list[str]:
    """
    Get function words for a given language code.

    Parameters
    ----------
    language : str — ISO 639-1 code ('fr', 'en')

    Returns
    -------
    list[str]

    Raises
    ------
    ValueError if language not supported
    """
    if language not in LANGUAGES:
        supported = ", ".join(LANGUAGES.keys())
        raise ValueError(
            f"Language '{language}' not supported. "
            f"Supported: {supported}. "
            "Pass a custom list via function_words= for other languages."
        )
    return LANGUAGES[language]
