"""Curated Ukrainian sentiment lexicon and zero-dependency sentiment analyzer."""

from __future__ import annotations

from ._core import (
    ApostropheStyle,
    ModifierKind,
    Polarity,
    SentimentLabel,
    SentimentMatch,
    SentimentResult,
    Tier,
    analyze_sentiment,
    get_lexicon,
    get_modifiers,
    get_phrases,
    get_words_by_polarity,
    list_categories,
    normalize_apostrophes,
    score_text,
    tokenize_ukrainian,
)

__version__ = "2.0.0"

__all__ = [
    "ApostropheStyle",
    "ModifierKind",
    "Polarity",
    "SentimentLabel",
    "SentimentMatch",
    "SentimentResult",
    "Tier",
    "__version__",
    "analyze_sentiment",
    "get_lexicon",
    "get_modifiers",
    "get_phrases",
    "get_words_by_polarity",
    "list_categories",
    "normalize_apostrophes",
    "score_text",
    "tokenize_ukrainian",
]
