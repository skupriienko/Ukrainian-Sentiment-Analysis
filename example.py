"""Usage examples for ukrainian-sentiment (analyzer API, raw CSV, and pandas)."""

from __future__ import annotations

import csv
import importlib
import importlib.util
from pathlib import Path
from typing import Any

from ukrainian_sentiment import (
    analyze_sentiment,
    get_lexicon,
    get_phrases,
    score_text,
)


def main() -> None:
    """Demonstrate rule-based sentiment scoring, modifiers, and raw CSV loading."""
    samples = [
        "Дуже гарна книга, щиро дякую!",
        "Це не поганий результат, все пройшло без проблем.",
        "На жаль, сервіс жахливий і доставка повільна.",
        "Товар не дуже гарний, проте без помилок.",
    ]

    for text in samples:
        res = analyze_sentiment(text)
        print(f"[{res.label:8s} score={res.score:5.2f}] {text}")

    # 1. Lexicon sizes and profanity toggle
    full_lex = get_lexicon("extended", include_profanity=True)
    clean_lex = get_lexicon("extended", include_profanity=False)
    print("Extended lexicon size:", len(full_lex), "(without profanity:", len(clean_lex), ")")
    print("Multi-word sentiment phrases (with apostrophe variants):", len(get_phrases()))

    # 2. Loading BOM-free CSV directly
    csv_path = Path(__file__).resolve().parent / "sentiment_ua.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        legacy_entries = {row["word"]: int(row["pos_neg"]) for row in reader}
    print("Root sentiment_ua.csv entries:", len(legacy_entries))

    # 3. Optional pandas DataFrame integration
    if importlib.util.find_spec("pandas") is None:
        return

    pd: Any = importlib.import_module("pandas")
    df = pd.DataFrame({"review": samples})
    df["score"] = df["review"].apply(score_text)
    df["label"] = df["review"].apply(lambda txt: analyze_sentiment(txt).label)
    print("Pandas output:\n", df[["review", "score", "label"]])


if __name__ == "__main__":
    main()
