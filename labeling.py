"""Weak supervision module for heuristic-based spam labeling."""
import re
from typing import Dict, List, Tuple

import numpy as np

# =============================================================================
# REGEX PATTERNS (Pre-compiled for performance)
# =============================================================================

RE_PUNCTUATION = re.compile(r"[!?]")
RE_EMOJI = re.compile(r"[\U0001F600-\U0001F6FF]")
RE_REPEATED_CHAR = re.compile(r"(.)\1{2,}")
RE_URL = re.compile(r"https?://|www\.")


def count_spam_signals(text: str) -> int:
    """
    Count spam indicators in a text.

    Spam signals detected:
    - Text shorter than 5 words
    - Excessive punctuation (! or ?)
    - All uppercase text
    - Emoji presence
    - Repeated characters (e.g., "süpeeeer")
    - URL presence

    Args:
        text: Input text to analyze.

    Returns:
        Number of spam signals detected (0-6).
    """
    if not isinstance(text, str):
        return 0

    signals = 0
    words = text.split()
    word_count = len(words)

    if word_count < 5:
        signals += 1

    if len(RE_PUNCTUATION.findall(text)) > 3:
        signals += 1

    if text.isupper() and word_count > 1:
        signals += 1

    if RE_EMOJI.search(text):
        signals += 1

    if RE_REPEATED_CHAR.search(text):
        signals += 1

    if RE_URL.search(text):
        signals += 1

    return signals


def label_texts(texts: List[str], threshold: int = 1) -> Tuple[np.ndarray, Dict]:
    """
    Label texts as spam (1) or genuine (0) based on heuristic signals.

    Uses np.int8 for memory efficiency.

    Args:
        texts: List of texts to label.
        threshold: Minimum spam signals required to label as spam.

    Returns:
        Tuple of (labels array, statistics dictionary).
    """
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
    """
    Print weak supervision labeling statistics.

    Args:
        stats: Statistics dictionary from label_texts().
    """
    print("\n" + "=" * 50)
    print(" WEAK SUPERVISION LABELING RESULTS")
    print("=" * 50)
    print(f"Total texts:       {stats['total']:,}")
    print(f"Genuine (0):       {stats['genuine']:,} ({stats['genuine_pct']}%)")
    print(f"Spam (1):          {stats['spam']:,} ({stats['spam_pct']}%)")
    print("=" * 50)
