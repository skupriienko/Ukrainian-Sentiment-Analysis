"""Tests for the ukrainian_sentiment package and generated datasets."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_datasets  # noqa: E402
from ukrainian_sentiment import (  # noqa: E402
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


class SentimentDatasetAndAnalyzerTest(unittest.TestCase):
    """Validate dataset integrity, legacy compatibility, and analyzer behavior."""

    def test_build_script_check_passes(self) -> None:
        self.assertEqual(build_datasets.main(["--check"]), 0)

    def test_categories_disjoint_and_valid(self) -> None:
        categories = build_datasets.load_and_validate_categories(REPO_ROOT)
        self.assertEqual(tuple(categories.keys()), build_datasets.ALL_CATEGORIES)
        self.assertEqual(list_categories(), build_datasets.ALL_CATEGORIES)

        seen_unigrams: set[str] = set()
        for name in build_datasets.UNIGRAM_CATEGORIES:
            words = set(categories[name])
            self.assertFalse(seen_unigrams & words, f"Overlap in {name}: {seen_unigrams & words}")
            seen_unigrams |= words

        seen_phrases: set[str] = set()
        for name in build_datasets.PHRASE_CATEGORIES:
            phrases = set(categories[name])
            self.assertFalse(seen_phrases & phrases, f"Overlap in {name}: {seen_phrases & phrases}")
            seen_phrases |= phrases

    def test_legacy_csv_has_no_bom_and_matches_json(self) -> None:
        legacy_bytes = (REPO_ROOT / "sentiment_ua.csv").read_bytes()
        self.assertFalse(legacy_bytes.startswith(b"\xef\xbb\xbf"))

        json_data = json.loads((REPO_ROOT / "data" / "sentiment_ua.json").read_text("utf-8"))
        with (REPO_ROOT / "sentiment_ua.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=";"))

        self.assertEqual(list(rows[0].keys()), ["word", "pos_neg"])
        self.assertEqual(len(rows), json_data["counts"]["legacy_total"])

    def test_purged_calques_and_noise_absent(self) -> None:
        lexicon = get_lexicon("extended", include_phrases=True, apostrophe="ascii")
        purged = [
            "дуже",
            "лимон",
            "пацієнт",
            "син",
            "сік",
            "гліб",
            "сейф",
            "сюжет",
            "адвокат",
            "адвокати",
            "душові мішки",
            "цукрова шуба",
            "вентиляційний отвір",
            "щебетливый",
            "знайдибiда",
            "жид",
            "м`ясопродукт",
            "анти-",
        ]
        for term in purged:
            self.assertNotIn(term, lexicon, f"Purged term {term!r} should be absent")

    def test_added_words_and_polarity_fixes(self) -> None:
        core = get_lexicon("core", include_phrases=True, apostrophe="ascii")
        extended = get_lexicon("extended", include_phrases=True, apostrophe="ascii")

        for pos in ("гарний", "чудовий", "добре", "дякую", "перемога", "щебетливий", "мертво"):
            self.assertIn(pos, core)
        self.assertEqual(core["гарний"], 1)
        self.assertEqual(core["чудовий"], 1)
        self.assertEqual(core["дякую"], 1)
        self.assertEqual(core["перемога"], 1)
        self.assertEqual(core["щебетливий"], 1)
        self.assertEqual(core["мертво"], -1)
        self.assertEqual(core["відповідальність"], 1)

        for neg in ("поганий", "жахливий", "зрада", "ганьба", "знайдибіда", "з жахом"):
            self.assertEqual(core[neg], -1)

        self.assertEqual(extended["супер"], 1)
        self.assertEqual(extended["крінж"], -1)
        self.assertLess(len(core), len(extended))

    def test_apostrophe_variants_and_profanity_toggle(self) -> None:
        all_lex = get_lexicon("extended", apostrophe="all")
        self.assertIn("здоров'я", all_lex)
        self.assertIn("здоров\u2019я", all_lex)
        self.assertIn("здоров\u02bcя", all_lex)

        typo_only = get_lexicon("core", apostrophe="typographic")
        self.assertIn("здоров\u2019я", typo_only)
        self.assertNotIn("здоров'я", typo_only)

        self.assertEqual(normalize_apostrophes("здоров\u2019я і сім\u02bcя"), "здоров'я і сім'я")

        no_prof = get_lexicon("extended", include_profanity=False, apostrophe="ascii")
        self.assertNotIn("хуй", no_prof)
        self.assertIn("крінж", no_prof)

    def test_polarity_helpers_and_modifiers(self) -> None:
        pos_words = get_words_by_polarity("positive", tier="core", apostrophe="ascii")
        neg_words = get_words_by_polarity("negative", tier="core", apostrophe="ascii")
        self.assertIn("чудовий", pos_words)
        self.assertIn("жахливий", neg_words)
        self.assertFalse(pos_words & neg_words)

        self.assertIn("не", get_modifiers("negations"))
        self.assertIn("дуже", get_modifiers("intensifiers"))
        self.assertIn("трохи", get_modifiers("diminishers"))
        self.assertEqual(get_phrases(apostrophe="ascii")["на жаль"], -1)

    def test_analyze_sentiment_modifiers_and_phrases(self) -> None:
        self.assertEqual(
            tokenize_ukrainian("Дев’ять ідей, з-під землі!"),
            ["Дев’ять", "ідей", "з-під", "землі"],
        )

        pos = analyze_sentiment("Дуже гарна книга, щиро дякую!")
        self.assertEqual(pos.label, "positive")
        self.assertGreater(pos.score, score_text("Гарна книга"))

        negated_pos = analyze_sentiment("Це не гарний варіант")
        self.assertEqual(negated_pos.label, "negative")
        self.assertLess(negated_pos.score, 0.0)

        negated_neg = analyze_sentiment("Це не поганий результат")
        self.assertEqual(negated_neg.label, "positive")
        self.assertGreater(negated_neg.score, 0.0)

        neg_intens = analyze_sentiment("Не дуже гарний сервіс")
        self.assertEqual(neg_intens.label, "negative")
        self.assertLess(neg_intens.score, 0.0)

        diminished = analyze_sentiment("Трохи сумний фінал")
        self.assertEqual(diminished.label, "negative")
        self.assertAlmostEqual(diminished.score, -0.5)

        phrase_res = analyze_sentiment("Все пройшло без проблем.")
        self.assertEqual(phrase_res.label, "positive")
        self.assertEqual(phrase_res.matches[0].text, "без проблем")

        comma_barrier = analyze_sentiment("Ні, гарний варіант")
        self.assertEqual(comma_barrier.label, "positive")
        self.assertEqual(comma_barrier.score, 1.0)

        neutral = analyze_sentiment("Купили дві ручки та зошит.")
        self.assertEqual(neutral.label, "neutral")
        self.assertEqual(neutral.score, 0.0)

    def test_invalid_arguments_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            get_lexicon(cast(Any, "invalid"))
        with self.assertRaises(ValueError):
            get_lexicon(categories=["negations"])
        with self.assertRaises(ValueError):
            get_lexicon(apostrophe=cast(Any, "invalid"))
        with self.assertRaises(ValueError):
            get_words_by_polarity(cast(Any, "neutral"))
        with self.assertRaises(ValueError):
            get_modifiers(cast(Any, "unknown"))
        with self.assertRaises(ValueError):
            analyze_sentiment("тест", neutral_threshold=-1.0)


if __name__ == "__main__":
    unittest.main()
