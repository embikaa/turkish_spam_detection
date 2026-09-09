import re
from typing import Dict, List, Tuple

import numpy as np

RE_PUNCTUATION = re.compile(r"[!?]")
RE_EMOJI = re.compile(r"[\U0001F600-\U0001F6FF]")
RE_REPEATED_CHAR = re.compile(r"(.)\1{2,}")
RE_URL = re.compile(r"https?://|www\.")


def count_spam_signals(text: str) -> int:
    return len(get_spam_signals(text))


def get_spam_signals(text: str) -> List[str]:
    if not isinstance(text, str):
        return []

    triggered = []
    words = text.split()
    word_count = len(words)

    if word_count < 5:
        triggered.append("Text is too short less than 5 words")

    if len(RE_PUNCTUATION.findall(text)) > 3:
        triggered.append("Oversampling of Punctuation")

    if text.isupper() and word_count > 1:
        triggered.append("All text wrote uppercase")

    if RE_EMOJI.search(text):
        triggered.append("Comment have emoji")

    if RE_REPEATED_CHAR.search(text):
        triggered.append("Repeated words included")

    if RE_URL.search(text):
        triggered.append("Suspicious URL included")

    return triggered


def label_texts(texts: List[str], threshold: int = 1) -> Tuple[np.ndarray, Dict]:
    labels = np.array(
        [1 if count_spam_signals(t) >= threshold else 0 for t in texts],
        dtype=np.int8
    )

    spam_count = int(np.sum(labels))
    total = len(labels)
    genuine_count = total - spam_count

    spam_pct = round(spam_count / total * 100, 2) if total > 0 else 0.0
    genuine_pct = round(genuine_count / total * 100, 2) if total > 0 else 0.0

    stats = {
        "total": total,
        "genuine": genuine_count,
        "spam": spam_count,
        "genuine_pct": genuine_pct,
        "spam_pct": spam_pct
    }

    return labels, stats


def print_label_stats(stats: Dict) -> None:
    print("\n" + "=" * 50)
    print(" WEAK SUPERVISION LABELING RESULTS")
    print("=" * 50)
    print(f"Total texts:       {stats['total']:,}")
    print(f"Genuine (0):       {stats['genuine']:,} ({stats['genuine_pct']}%)")
    print(f"Spam (1):          {stats['spam']:,} ({stats['spam_pct']}%)")
    print("=" * 50)
