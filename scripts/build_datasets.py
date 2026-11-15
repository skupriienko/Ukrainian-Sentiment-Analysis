"""Deterministically build and validate Ukrainian sentiment lexicon files."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

VERSION = "2.0.0"
LICENSE = "CC-BY-SA-4.0"

CORE_SENTIMENT_CATEGORIES: dict[str, int] = {
    "positive_core": 1,
    "negative_core": -1,
}

EXTENDED_ONLY_SENTIMENT_CATEGORIES: dict[str, int] = {
    "positive_colloquial": 1,
    "negative_colloquial": -1,
    "negative_profanity": -1,
}

SENTIMENT_UNIGRAM_CATEGORIES: dict[str, int] = {
    **CORE_SENTIMENT_CATEGORIES,
    **EXTENDED_ONLY_SENTIMENT_CATEGORIES,
}

PHRASE_CATEGORIES: dict[str, int] = {
    "phrases_positive": 1,
    "phrases_negative": -1,
}

MODIFIER_CATEGORIES: tuple[str, ...] = (
    "negations",
    "intensifiers",
    "diminishers",
)

UNIGRAM_CATEGORIES: tuple[str, ...] = tuple(SENTIMENT_UNIGRAM_CATEGORIES) + MODIFIER_CATEGORIES
ALL_CATEGORIES: tuple[str, ...] = (
    tuple(SENTIMENT_UNIGRAM_CATEGORIES)
    + tuple(PHRASE_CATEGORIES)
    + MODIFIER_CATEGORIES
)

_UK_UNIGRAM_RE = re.compile(r"^[а-яґєії]+(?:['-][а-яґєії]+)*$")
_UK_PHRASE_RE = re.compile(
    r"^[а-яґєії]+(?:['-][а-яґєії]+)*(?: [а-яґєії]+(?:['-][а-яґєії]+)*)+$"
)


def validate_category_file(path: Path, *, is_phrase: bool) -> list[str]:
    """Read and strictly validate a category source file."""
    if not path.is_file():
        raise ValueError(f"Missing category file: {path}")

    raw_bytes = path.read_bytes()
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path.name}: UTF-8 BOM is not allowed")

    raw = raw_bytes.decode("utf-8")
    if not raw.endswith("\n"):
        raise ValueError(f"{path.name}: must end with a trailing newline")

    lines = raw.splitlines()
    if not lines:
        raise ValueError(f"{path.name}: category file must not be empty")

    pattern = _UK_PHRASE_RE if is_phrase else _UK_UNIGRAM_RE
    seen: set[str] = set()
    validated: list[str] = []

    for idx, entry in enumerate(lines, start=1):
        if entry == "":
            raise ValueError(f"{path.name}:{idx}: blank line is not allowed")
        if entry != entry.strip():
            raise ValueError(f"{path.name}:{idx}: leading/trailing whitespace in {entry!r}")
        if unicodedata.normalize("NFC", entry) != entry:
            raise ValueError(f"{path.name}:{idx}: entry {entry!r} is not NFC-normalized")
        if entry != entry.lower():
            raise ValueError(f"{path.name}:{idx}: entry {entry!r} must be lowercase")
        if entry == "ь":
            raise ValueError(f"{path.name}:{idx}: standalone 'ь' is not allowed")
        if not pattern.match(entry):
            raise ValueError(f"{path.name}:{idx}: invalid characters or spacing in {entry!r}")
        if entry in seen:
            raise ValueError(f"{path.name}:{idx}: duplicate entry {entry!r}")
        seen.add(entry)
        validated.append(entry)

    if validated != sorted(validated):
        raise ValueError(f"{path.name}: entries must be sorted in ascending Unicode order")

    return validated


def load_and_validate_categories(repo_root: Path) -> dict[str, list[str]]:
    """Load all category files and enforce zero overlap within unigrams and phrases."""
    cat_dir = repo_root / "data" / "categories"
    found_files = {p.stem for p in cat_dir.glob("*.txt")}
    expected_files = set(ALL_CATEGORIES)
    if found_files != expected_files:
        extra = sorted(found_files - expected_files)
        missing = sorted(expected_files - found_files)
        raise ValueError(f"Category directory mismatch: missing={missing}, unexpected={extra}")

    categories: dict[str, list[str]] = {}
    seen_unigrams: dict[str, str] = {}
    seen_phrases: dict[str, str] = {}

    for name in ALL_CATEGORIES:
        is_phrase = name in PHRASE_CATEGORIES
        entries = validate_category_file(cat_dir / f"{name}.txt", is_phrase=is_phrase)
        registry = seen_phrases if is_phrase else seen_unigrams
        kind = "phrase" if is_phrase else "unigram"
        for entry in entries:
            if entry in registry:
                raise ValueError(
                    f"Duplicate {kind} {entry!r} in {name}.txt and {registry[entry]}.txt"
                )
            registry[entry] = name
        categories[name] = entries

    return categories


def _format_pairs(header: str, pairs: list[tuple[str, int]], sep: str) -> str:
    rows = [f"{term}{sep}{score}" for term, score in sorted(pairs, key=lambda item: item[0])]
    return "\n".join([header, *rows]) + "\n"


def build_artifacts(repo_root: Path) -> dict[Path, str]:
    """Compute expected generated file contents keyed by repository-relative Path."""
    categories = load_and_validate_categories(repo_root)

    core_pairs: list[tuple[str, int]] = [
        (word, score)
        for name, score in CORE_SENTIMENT_CATEGORIES.items()
        for word in categories[name]
    ]
    extended_only_pairs: list[tuple[str, int]] = [
        (word, score)
        for name, score in EXTENDED_ONLY_SENTIMENT_CATEGORIES.items()
        for word in categories[name]
    ]
    extended_pairs = core_pairs + extended_only_pairs
    phrase_pairs: list[tuple[str, int]] = [
        (phrase, score)
        for name, score in PHRASE_CATEGORIES.items()
        for phrase in categories[name]
    ]
    legacy_pairs = extended_pairs + phrase_pairs

    modifiers_count = sum(len(categories[name]) for name in MODIFIER_CATEGORIES)
    payload = {
        "version": VERSION,
        "license": LICENSE,
        "counts": {
            "core": len(core_pairs),
            "extended": len(extended_pairs),
            "phrases": len(phrase_pairs),
            "modifiers": modifiers_count,
            "legacy_total": len(legacy_pairs),
            "categories": {name: len(entries) for name, entries in categories.items()},
        },
        "categories": categories,
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    return {
        repo_root / "data" / "sentiment_core.csv": _format_pairs("word,score", core_pairs, ","),
        repo_root / "data" / "sentiment_extended.csv": _format_pairs(
            "word,score", extended_pairs, ","
        ),
        repo_root / "data" / "phrases_ua.csv": _format_pairs("phrase,score", phrase_pairs, ","),
        repo_root / "data" / "sentiment_ua.json": json_text,
        repo_root / "src" / "ukrainian_sentiment" / "data" / "sentiment_ua.json": json_text,
        repo_root / "sentiment_ua.csv": _format_pairs("word;pos_neg", legacy_pairs, ";"),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for building or checking generated sentiment datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that all generated files are up to date without writing changes.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    artifacts = build_artifacts(repo_root)

    if args.check:
        mismatches: list[Path] = []
        for path, expected in artifacts.items():
            if not path.is_file() or path.read_bytes() != expected.encode("utf-8"):
                mismatches.append(path.relative_to(repo_root))
        if mismatches:
            print("Generated files are out of date:", ", ".join(str(p) for p in mismatches))
            return 1
        print("All sentiment datasets and JSON artifacts are valid and up to date.")
        return 0

    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))

    print(f"Built {len(artifacts)} sentiment distribution files successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
