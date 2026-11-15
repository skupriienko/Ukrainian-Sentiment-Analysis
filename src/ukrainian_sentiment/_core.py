"""Core lexicon loaders, normalization helpers, and rule-based Ukrainian sentiment scoring."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Literal, TypedDict

Tier = Literal["core", "extended"]
Polarity = Literal["positive", "negative"]
SentimentLabel = Literal["positive", "negative", "neutral"]
ApostropheStyle = Literal["ascii", "typographic", "modifier", "all"]
ModifierKind = Literal["negations", "intensifiers", "diminishers"]

_APOSTROPHES: tuple[str, ...] = ("'", "\u2019", "\u02bc")
_APOSTROPHE_MAP: dict[ApostropheStyle, str] = {
    "ascii": "'",
    "typographic": "\u2019",
    "modifier": "\u02bc",
}
_TOKEN_RE = re.compile(
    r"[0-9A-Za-zА-Яа-яҐґЄєІіЇї]+(?:['\u2019\u02bc-][0-9A-Za-zА-Яа-яҐґЄєІіЇї]+)*"
)

_CATEGORY_POLARITY: dict[str, int] = {
    "positive_core": 1,
    "negative_core": -1,
    "positive_colloquial": 1,
    "negative_colloquial": -1,
    "negative_profanity": -1,
    "phrases_positive": 1,
    "phrases_negative": -1,
}
_CORE_CATEGORIES: tuple[str, ...] = ("positive_core", "negative_core")
_EXTENDED_COLLOQUIAL_CATEGORIES: tuple[str, ...] = (
    "positive_core",
    "negative_core",
    "positive_colloquial",
    "negative_colloquial",
)
_PHRASE_CATEGORIES: tuple[str, ...] = ("phrases_positive", "phrases_negative")

# Conservative Ukrainian suffix rewrites to candidate base forms (adjectives, nouns, adverbs).
_INFLECTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ого", ("ий", "ій")),
    ("ому", ("ий", "ій")),
    ("ему", ("ій", "ий")),
    ("ою", ("ий", "а", "")),
    ("ої", ("ий", "а")),
    ("ій", ("ий", "а", "я", "")),
    ("им", ("ий", "")),
    ("ім", ("ій", "")),
    ("их", ("ий", "ій")),
    ("ої", ("а",)),
    ("ами", ("а", "и", "")),
    ("ями", ("я", "ь", "")),
    ("ах", ("а", "и", "")),
    ("ях", ("я", "ь", "")),
    ("ам", ("а", "и", "")),
    ("ям", ("я", "ь", "")),
    ("ів", ("", "а")),
    ("тю", ("ть",)),
    ("істю", ("ість",)),
    ("ості", ("ість",)),
    ("а", ("ий", "")),
    ("я", ("ій", "ь", "")),
    ("е", ("ий", "ій", "")),
    ("і", ("ий", "ій", "а", "я", "ь", "")),
    ("у", ("ий", "а", "")),
    ("ю", ("ій", "я", "ь", "")),
    ("и", ("а", "")),
    ("о", ("ий", "ій")),
)


class _CountsPayload(TypedDict):
    core: int
    extended: int
    phrases: int
    modifiers: int
    legacy_total: int
    categories: dict[str, int]


class _DatasetPayload(TypedDict):
    version: str
    license: str
    counts: _CountsPayload
    categories: dict[str, list[str]]


@dataclass(frozen=True)
class SentimentMatch:
    """Represents a single matched word or phrase and its modifier-adjusted contribution."""

    text: str
    base_score: int
    adjusted_score: float
    modifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class SentimentResult:
    """Aggregated sentiment analysis result for an input text."""

    score: float
    comparative: float
    label: SentimentLabel
    positive_count: int
    negative_count: int
    token_count: int
    matches: tuple[SentimentMatch, ...]


@lru_cache(maxsize=1)
def _load_dataset() -> _DatasetPayload:
    raw = (
        files("ukrainian_sentiment")
        .joinpath("data")
        .joinpath("sentiment_ua.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


def normalize_apostrophes(text: str, target: str = "'") -> str:
    """Normalize Ukrainian apostrophe variants (', ’, ʼ) and NFC composition."""
    normalized = unicodedata.normalize("NFC", text)
    for variant in _APOSTROPHES:
        if variant != target:
            normalized = normalized.replace(variant, target)
    return normalized


def _validate_apostrophe_style(style: ApostropheStyle) -> None:
    if style != "all" and style not in _APOSTROPHE_MAP:
        raise ValueError(
            f"Unsupported apostrophe style {style!r}; "
            "expected 'ascii', 'typographic', 'modifier', or 'all'."
        )


def _expand_word_variants(word: str, style: ApostropheStyle) -> tuple[str, ...]:
    if "'" not in word:
        return (word,)
    if style == "all":
        return tuple(word.replace("'", variant) for variant in _APOSTROPHES)
    replacement = _APOSTROPHE_MAP[style]
    return (word if replacement == "'" else word.replace("'", replacement),)


def _expand_apostrophes_set(words: Iterable[str], style: ApostropheStyle) -> frozenset[str]:
    _validate_apostrophe_style(style)
    if style == "ascii":
        return frozenset(words)
    expanded: set[str] = set()
    for word in words:
        expanded.update(_expand_word_variants(word, style))
    return frozenset(expanded)


def _expand_apostrophes_dict(
    pairs: Iterable[tuple[str, int]],
    style: ApostropheStyle,
) -> dict[str, int]:
    _validate_apostrophe_style(style)
    if style == "ascii":
        return dict(pairs)
    expanded: dict[str, int] = {}
    for word, score in pairs:
        for variant in _expand_word_variants(word, style):
            expanded[variant] = score
    return expanded


def list_categories() -> tuple[str, ...]:
    """Return all available lexicon and modifier category names in deterministic order."""
    return tuple(_load_dataset()["categories"].keys())


def get_modifiers(kind: ModifierKind, *, apostrophe: ApostropheStyle = "all") -> frozenset[str]:
    """Return a frozen set of negation, intensifier, or diminisher tokens."""
    if kind not in ("negations", "intensifiers", "diminishers"):
        invalid_kind: object = kind
        raise ValueError(
            f"Unsupported modifier kind {invalid_kind!r}; "
            "expected 'negations', 'intensifiers', or 'diminishers'."
        )
    entries = _load_dataset()["categories"][kind]
    return _expand_apostrophes_set(entries, apostrophe)


def get_phrases(*, apostrophe: ApostropheStyle = "all") -> dict[str, int]:
    """Return multi-word Ukrainian sentiment phrases mapped to +1 or -1."""
    categories = _load_dataset()["categories"]
    pairs = [
        (phrase, _CATEGORY_POLARITY[name])
        for name in _PHRASE_CATEGORIES
        for phrase in categories[name]
    ]
    return _expand_apostrophes_dict(pairs, apostrophe)


def _select_unigram_categories(tier: Tier, include_profanity: bool) -> tuple[str, ...]:
    if tier == "core":
        return _CORE_CATEGORIES
    if tier == "extended":
        if include_profanity:
            return (*_EXTENDED_COLLOQUIAL_CATEGORIES, "negative_profanity")
        return _EXTENDED_COLLOQUIAL_CATEGORIES
    invalid_tier: object = tier
    raise ValueError(f"Unsupported tier {invalid_tier!r}; expected 'core' or 'extended'.")


def get_lexicon(
    tier: Tier = "extended",
    *,
    include_phrases: bool = False,
    include_profanity: bool = True,
    categories: Iterable[str] | None = None,
    apostrophe: ApostropheStyle = "all",
) -> dict[str, int]:
    """Return a mapping of Ukrainian sentiment terms to polarity (+1 or -1)."""
    available = _load_dataset()["categories"]
    selected_names: list[str]

    if categories is not None:
        selected_names = []
        for name in categories:
            if name not in _CATEGORY_POLARITY:
                valid = ", ".join(_CATEGORY_POLARITY.keys())
                raise ValueError(
                    f"Unknown or non-scored category {name!r}; expected one of: {valid}"
                )
            selected_names.append(name)
        if not include_profanity:
            selected_names = [name for name in selected_names if name != "negative_profanity"]
    else:
        selected_names = list(_select_unigram_categories(tier, include_profanity))

    if include_phrases:
        for phrase_cat in _PHRASE_CATEGORIES:
            if phrase_cat not in selected_names:
                selected_names.append(phrase_cat)

    pairs = [
        (entry, _CATEGORY_POLARITY[name])
        for name in selected_names
        for entry in available[name]
    ]
    return _expand_apostrophes_dict(pairs, apostrophe)


def get_words_by_polarity(
    polarity: Polarity,
    *,
    tier: Tier = "extended",
    include_phrases: bool = False,
    include_profanity: bool = True,
    apostrophe: ApostropheStyle = "all",
) -> frozenset[str]:
    """Return a frozen set of positive or negative sentiment terms."""
    if polarity == "positive":
        target = 1
    elif polarity == "negative":
        target = -1
    else:
        invalid_polarity: object = polarity
        raise ValueError(
            f"Unsupported polarity {invalid_polarity!r}; expected 'positive' or 'negative'."
        )

    lexicon = get_lexicon(
        tier,
        include_phrases=include_phrases,
        include_profanity=include_profanity,
        apostrophe=apostrophe,
    )
    return frozenset(word for word, score in lexicon.items() if score == target)


def tokenize_ukrainian(text: str) -> list[str]:
    """Tokenize Ukrainian text while preserving internal apostrophes and hyphens."""
    return _TOKEN_RE.findall(unicodedata.normalize("NFC", text))


def _resolve_inflected_score(token: str, lexicon: Mapping[str, int]) -> int | None:
    if token in lexicon:
        return lexicon[token]
    if len(token) < 5:
        return None

    candidate_scores: set[int] = set()
    for suffix, replacements in _INFLECTION_RULES:
        if not token.endswith(suffix):
            continue
        stem = token[: len(token) - len(suffix)]
        if len(stem) < 3:
            continue
        for replacement in replacements:
            candidate = stem + replacement
            score = lexicon.get(candidate)
            if score is not None:
                candidate_scores.add(score)
                if len(candidate_scores) > 1:
                    return None

    if len(candidate_scores) == 1:
        return next(iter(candidate_scores))
    return None


@lru_cache(maxsize=16)
def _analyzer_resources(
    tier: Tier,
    include_phrases: bool,
    include_profanity: bool,
) -> tuple[
    dict[str, int],
    dict[tuple[str, ...], int],
    int,
    frozenset[str],
    frozenset[str],
    frozenset[str],
]:
    unigrams = get_lexicon(
        tier,
        include_phrases=False,
        include_profanity=include_profanity,
        apostrophe="ascii",
    )
    phrase_tuples: dict[tuple[str, ...], int] = {}
    max_phrase_len = 0
    if include_phrases:
        for phrase, score in get_phrases(apostrophe="ascii").items():
            parts = tuple(phrase.split(" "))
            phrase_tuples[parts] = score
            if len(parts) > max_phrase_len:
                max_phrase_len = len(parts)

    negations = get_modifiers("negations", apostrophe="ascii")
    intensifiers = get_modifiers("intensifiers", apostrophe="ascii")
    diminishers = get_modifiers("diminishers", apostrophe="ascii")
    return unigrams, phrase_tuples, max_phrase_len, negations, intensifiers, diminishers


def _collect_adjacent_modifiers(
    token_matches: list[re.Match[str]],
    tokens_lower: list[str],
    consumed: list[bool],
    normalized_text: str,
    target_idx: int,
    negations: frozenset[str],
    intensifiers: frozenset[str],
    diminishers: frozenset[str],
) -> tuple[str, ...]:
    collected: list[str] = []
    curr_idx = target_idx
    for _ in range(2):
        prev_idx = curr_idx - 1
        if prev_idx < 0 or consumed[prev_idx]:
            break
        gap = normalized_text[token_matches[prev_idx].end() : token_matches[curr_idx].start()]
        if gap.strip():
            break
        word = tokens_lower[prev_idx]
        if word not in negations and word not in intensifiers and word not in diminishers:
            break
        collected.append(word)
        curr_idx = prev_idx
    collected.reverse()
    return tuple(collected)


def _apply_modifiers(
    base_score: int,
    modifiers: tuple[str, ...],
    negations: frozenset[str],
    intensifiers: frozenset[str],
    diminishers: frozenset[str],
) -> float:
    if not modifiers:
        return float(base_score)

    if len(modifiers) == 2:
        first, second = modifiers
        if first in negations and second in intensifiers:
            return round(-0.5 * base_score, 4)
        if first in negations and second in diminishers:
            return round(-0.5 * base_score, 4)
        if first in intensifiers and second in negations:
            negated = -0.5 * base_score if base_score < 0 else -0.8 * base_score
            return round(1.5 * negated, 4)
        if first in diminishers and second in negations:
            negated = -0.5 * base_score if base_score < 0 else -0.8 * base_score
            return round(0.5 * negated, 4)
        active = second
    else:
        active = modifiers[0]

    if active in negations:
        factor = -0.5 if base_score < 0 else -0.8
        return round(factor * base_score, 4)
    if active in intensifiers:
        return round(1.5 * base_score, 4)
    if active in diminishers:
        return round(0.5 * base_score, 4)
    return float(base_score)


def analyze_sentiment(
    text: str,
    *,
    tier: Tier = "extended",
    include_phrases: bool = True,
    include_profanity: bool = True,
    normalize_inflections: bool = True,
    neutral_threshold: float = 0.0,
) -> SentimentResult:
    """Compute rule-based sentiment score with phrase, negation, and intensifier handling."""
    if neutral_threshold < 0:
        raise ValueError("neutral_threshold must be non-negative.")

    normalized_text = normalize_apostrophes(text, target="'")
    token_matches = list(_TOKEN_RE.finditer(normalized_text))
    tokens_lower = [m.group(0).lower() for m in token_matches]
    token_count = len(tokens_lower)

    if token_count == 0:
        return SentimentResult(
            score=0.0,
            comparative=0.0,
            label="neutral",
            positive_count=0,
            negative_count=0,
            token_count=0,
            matches=(),
        )

    unigrams, phrase_tuples, max_phrase_len, negations, intensifiers, diminishers = (
        _analyzer_resources(tier, include_phrases, include_profanity)
    )

    consumed = [False] * token_count
    found: list[tuple[int, SentimentMatch]] = []

    # 1. Greedy longest-first phrase pass (stops across clause punctuation)
    if include_phrases and max_phrase_len >= 2:
        idx = 0
        while idx < token_count:
            matched_len = 0
            matched_score = 0
            max_span = min(max_phrase_len, token_count - idx)
            for span in range(max_span, 1, -1):
                end_idx = idx + span
                contiguous = all(
                    not normalized_text[
                        token_matches[k].end() : token_matches[k + 1].start()
                    ].strip()
                    for k in range(idx, end_idx - 1)
                )
                if not contiguous:
                    continue
                candidate = tuple(tokens_lower[idx:end_idx])
                if candidate in phrase_tuples:
                    matched_len = span
                    matched_score = phrase_tuples[candidate]
                    break

            if matched_len:
                modifiers = _collect_adjacent_modifiers(
                    token_matches,
                    tokens_lower,
                    consumed,
                    normalized_text,
                    idx,
                    negations,
                    intensifiers,
                    diminishers,
                )
                for k in range(idx, idx + matched_len):
                    consumed[k] = True
                adjusted = _apply_modifiers(
                    matched_score, modifiers, negations, intensifiers, diminishers
                )
                found.append(
                    (
                        idx,
                        SentimentMatch(
                            text=" ".join(tokens_lower[idx : idx + matched_len]),
                            base_score=matched_score,
                            adjusted_score=adjusted,
                            modifiers=modifiers,
                        ),
                    )
                )
                idx += matched_len
            else:
                idx += 1

    # 2. Unigram pass with optional Ukrainian inflection fallback
    for idx, word in enumerate(tokens_lower):
        if consumed[idx]:
            continue
        if word in negations or word in intensifiers or word in diminishers:
            continue

        base_score = (
            _resolve_inflected_score(word, unigrams)
            if normalize_inflections
            else unigrams.get(word)
        )
        if base_score is None:
            continue

        modifiers = _collect_adjacent_modifiers(
            token_matches,
            tokens_lower,
            consumed,
            normalized_text,
            idx,
            negations,
            intensifiers,
            diminishers,
        )
        consumed[idx] = True
        adjusted = _apply_modifiers(base_score, modifiers, negations, intensifiers, diminishers)
        found.append(
            (
                idx,
                SentimentMatch(
                    text=word,
                    base_score=base_score,
                    adjusted_score=adjusted,
                    modifiers=modifiers,
                ),
            )
        )

    found.sort(key=lambda item: item[0])
    matches = tuple(match for _, match in found)
    total_score = round(sum(m.adjusted_score for m in matches), 4)
    comparative = round(total_score / token_count, 4)
    positive_count = sum(1 for m in matches if m.adjusted_score > 0)
    negative_count = sum(1 for m in matches if m.adjusted_score < 0)

    if total_score > neutral_threshold:
        label: SentimentLabel = "positive"
    elif total_score < -neutral_threshold:
        label = "negative"
    else:
        label = "neutral"

    return SentimentResult(
        score=total_score,
        comparative=comparative,
        label=label,
        positive_count=positive_count,
        negative_count=negative_count,
        token_count=token_count,
        matches=matches,
    )


def score_text(
    text: str,
    *,
    tier: Tier = "extended",
    include_phrases: bool = True,
    include_profanity: bool = True,
    normalize_inflections: bool = True,
) -> float:
    """Return the aggregated sentiment score for ``text``."""
    return analyze_sentiment(
        text,
        tier=tier,
        include_phrases=include_phrases,
        include_profanity=include_profanity,
        normalize_inflections=normalize_inflections,
    ).score
