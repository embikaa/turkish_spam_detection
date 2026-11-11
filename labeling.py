"""labeling.py — rule-based weak labeling of review texts (Bulgu 1 / 2).

The dataset's labels are produced by the heuristic rules below, not by
human annotation. The referee report's consequences for this module:

  * Bulgu 1: the rules are weak supervision. Rule coverage, co-occurrence
    and multiple-rule statistics (get_labeling_diagnostics) make the
    labeling process auditable, as the report's LFAnalysis-style reporting
    demands. label_matrix exposes the binary label-function matrix that
    feeds those statistics and the control experiment in which the rules'
    input signals (punctuation, case, emoji, URLs) are masked from the
    model.
  * Bulgu 2: labels and rules coincide by construction, so the rules
    evaluated as a classifier form the circularity ceiling; main.py reports
    that ceiling as a number. No edit below can remove the circularity —
    the gold-standard validation the report asks for is an external,
    human-annotated dataset and lives outside this module.
"""
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple, Any

import numpy as np


# ====
# 1. REGULAR EXPRESSIONS
# ====

# H2: Consecutive repeated punctuation
# Examples: !!, ???, ....
RE_PUNCTUATION = re.compile(
    r"!{3,}|\.{4,}"
)

# H5: Three or more consecutive repetitions of the same letter.
# Restricted to letters, so "...", "---" or "000" do not match.
# Examples: çooook, harikaaaa, güzeeeel
RE_REPEATED_CHAR = re.compile(
    r"([^\W\d_])\1{2,}",
    flags=re.IGNORECASE
)

# URLs: http(s), www, and common bare domain patterns
RE_URL = re.compile(
    r"(?:https?://|www\.)\S+|"
    r"\b[\w-]+\.(?:com|net|org|edu|gov|tr|com\.tr|net\.tr|org\.tr)"
    r"(?:/\S*)?",
    flags=re.IGNORECASE
)

# Emoji and supplementary pictographic ranges.
# This is a practical heuristic, not a complete Unicode emoji parser.
RE_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]"
)


# ====
# 2. RULE GROUPS
# ====

# A single strong rule is enough to label a review as spam.
STRONG_RULES = frozenset({
    "Unusually Long Review",
    "Excessive Uppercase",
    "External URL Included",
})

WEAK_RULES = frozenset({
    "Short Review",
    "Repeated Characters Included",
    "Emoji Included",
    "Repeated Punctuation",
})

# Canonical rule order, used by label_matrix columns.
RULE_NAMES = (
    "Unusually Long Review",
    "Repeated Punctuation",
    "Excessive Uppercase",
    "External URL Included",
    "Repeated Characters Included",
    "Emoji Included",
    "Short Review",
)

# Default thresholds (shared by both public functions).
DEFAULT_MIN_LENGTH = 15
DEFAULT_MAX_LENGTH = 1500
DEFAULT_UPPERCASE_RATIO = 0.50
DEFAULT_UPPERCASE_MIN_CHARS = 10
DEFAULT_MIN_WEAK_RULES = 2


# ====
# 3. LABELING FUNCTION
# ====

def labeling_function(
    text: str,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    uppercase_ratio_threshold: float = DEFAULT_UPPERCASE_RATIO,
    uppercase_min_chars: int = DEFAULT_UPPERCASE_MIN_CHARS,
    min_weak_rules: int = DEFAULT_MIN_WEAK_RULES,
) -> Tuple[int, List[str]]:
    """
    Assigns a weak label to a review using heuristic rules.

    Labels:
        0 = Genuine
        1 = Spam

    Decision:
        Spam if at least one STRONG rule fires, or if at least
        `min_weak_rules` WEAK rules fire together.

    Returns:
        label: Integer weak label (0 or 1).
        triggered_rules: List of triggered heuristic names.
    """

    if not isinstance(text, str):
        text = ""

    text = text.strip()

    triggered_rules: List[str] = []

    if not text:
        return 0, triggered_rules

    # H1: Unusually long review
    if len(text) > max_length:
        triggered_rules.append("Unusually Long Review")

    # H2: Repeated punctuation
    if RE_PUNCTUATION.search(text):
        triggered_rules.append("Repeated Punctuation")

    # H3: Excessive uppercase (only with enough letters)
    letters = [char for char in text if char.isalpha()]

    if len(letters) >= uppercase_min_chars:
        uppercase_count = sum(1 for char in letters if char.isupper())

        if uppercase_count / len(letters) >= uppercase_ratio_threshold:
            triggered_rules.append("Excessive Uppercase")

    # H4: URL or external link
    if RE_URL.search(text):
        triggered_rules.append("External URL Included")

    # H5: Repeated letters (weak)
    if RE_REPEATED_CHAR.search(text):
        triggered_rules.append("Repeated Characters Included")

    # H6: Emoji usage (weak)
    if RE_EMOJI.search(text):
        triggered_rules.append("Emoji Included")

    # H7: Unusually short review (weak)
    if len(text) < min_length:
        triggered_rules.append("Short Review")

    strong_hits = sum(1 for rule in triggered_rules if rule in STRONG_RULES)
    weak_hits = sum(1 for rule in triggered_rules if rule in WEAK_RULES)

    label = 1 if (strong_hits >= 1 or weak_hits >= min_weak_rules) else 0

    return label, triggered_rules


# ====
# 4. DATASET LABELING
# ====

def label_texts(
    texts: List[str],
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    uppercase_ratio_threshold: float = DEFAULT_UPPERCASE_RATIO,
    uppercase_min_chars: int = DEFAULT_UPPERCASE_MIN_CHARS,
    min_weak_rules: int = DEFAULT_MIN_WEAK_RULES,
) -> Tuple[np.ndarray, List[List[str]]]:
    """
    Applies heuristic labeling functions to a collection
    of review texts.

    Returns:
        labels: NumPy array containing weak labels.
        all_triggered_rules: Triggered rules for each review.
    """

    labels = []
    all_triggered_rules = []

    for text in texts:
        label, triggered = labeling_function(
            text=text,
            min_length=min_length,
            max_length=max_length,
            uppercase_ratio_threshold=uppercase_ratio_threshold,
            uppercase_min_chars=uppercase_min_chars,
            min_weak_rules=min_weak_rules,
        )

        labels.append(label)
        all_triggered_rules.append(triggered)

    return np.asarray(labels, dtype=np.int64), all_triggered_rules


def label_matrix(texts: List[str]) -> np.ndarray:
    """Binary label-function (LF) matrix, shape (n_texts, n_rules).

    Bulgu 1: column j is the binary output of rule j (column order is
    RULE_NAMES). This is the input for LFAnalysis-style coverage,
    polarity and overlap reporting and for the control experiment in
    which the model is trained on text whose rule inputs have been
    masked (see preprocessing.clean_for_tfidf for the masked branch).
    """
    matrix = np.zeros((len(texts), len(RULE_NAMES)), dtype=np.int8)

    for i, text in enumerate(texts):
        _, triggered = labeling_function(text)
        for rule in triggered:
            matrix[i, RULE_NAMES.index(rule)] = 1

    return matrix


# ====
# 5. LABELING DIAGNOSTICS
# ====

def get_labeling_diagnostics(
    labels: np.ndarray,
    all_triggered_rules: List[List[str]],
) -> Dict[str, Any]:
    """
    Generates descriptive diagnostics for the weak-labeling process.

    Includes:
        - Class distribution
        - Individual rule coverage (count, percentage, reviews flagged
          only by this function, share of spam labels)
        - Rule co-occurrence
        - Number of reviews with multiple triggered rules

    These diagnostics describe heuristic behavior.
    They do not establish label validity against human judgments.
    """

    labels = np.asarray(labels)

    total_reviews = len(labels)

    if total_reviews != len(all_triggered_rules):
        raise ValueError(
            "The number of labels and rule lists must be identical."
        )

    class_counts = Counter(labels.tolist())

    rule_counts = Counter(
        rule
        for triggered in all_triggered_rules
        for rule in set(triggered)
    )

    rule_cooccurrence = Counter()

    for triggered in all_triggered_rules:
        unique_rules = sorted(set(triggered))

        for rule_a, rule_b in combinations(unique_rules, 2):
            rule_cooccurrence[(rule_a, rule_b)] += 1

    multiple_rule_reviews = sum(
        1 for triggered in all_triggered_rules
        if len(set(triggered)) > 1
    )

    # Reviews flagged by exactly this function and no other
    only_this_function = Counter(
        triggered[0]
        for triggered in all_triggered_rules
        if len(set(triggered)) == 1
    )

    # How often each function fires among the final spam labels
    spam_idx = np.flatnonzero(labels == 1)
    fires_in_spam = Counter(
        rule
        for i in spam_idx
        for rule in set(all_triggered_rules[i])
    )
    total_spam = int(class_counts.get(1, 0))

    rule_coverage = {}

    for rule, count in rule_counts.items():
        in_spam = fires_in_spam.get(rule, 0)
        rule_coverage[rule] = {
            "count": count,
            "percentage": (
                round((count / total_reviews) * 100, 2)
                if total_reviews else 0.0
            ),
            "only_this_function": only_this_function.get(rule, 0),
            "share_of_spam_labels": (
                round(in_spam / total_spam * 100, 1)
                if total_spam else 0.0
            ),
        }

    diagnostics = {
        "total_reviews": total_reviews,
        "class_distribution": {
            "genuine": class_counts.get(0, 0),
            "spam": class_counts.get(1, 0),
        },
        "class_percentages": {
            "genuine": round(
                class_counts.get(0, 0) / total_reviews * 100, 2
            ) if total_reviews else 0.0,
            "spam": round(
                class_counts.get(1, 0) / total_reviews * 100, 2
            ) if total_reviews else 0.0,
        },
        "rule_coverage": rule_coverage,
        "rule_cooccurrence": {
            f"{rule_a} + {rule_b}": count
            for (rule_a, rule_b), count
            in rule_cooccurrence.items()
        },
        "multiple_rule_reviews": multiple_rule_reviews,
        "multiple_rule_percentage": round(
            multiple_rule_reviews / total_reviews * 100, 2
        ) if total_reviews else 0.0,
    }

    return diagnostics


# ====
# 6. OPTIONAL DIAGNOSTIC REPORT
# ====

def print_labeling_diagnostics(
    diagnostics: Dict[str, Any]
) -> None:
    """
    Prints a readable summary of weak-labeling diagnostics.
    """

    print("\n" + "=" * 60)
    print("WEAK LABELING DIAGNOSTICS")
    print("=" * 60)

    print(f"Total reviews: {diagnostics['total_reviews']}")

    print("\nClass distribution:")
    for class_name, count in diagnostics["class_distribution"].items():
        percentage = diagnostics["class_percentages"][class_name]

        print(
            f"  {class_name.capitalize()}: "
            f"{count:,} ({percentage:.2f}%)"
        )

    print("\nIndividual rule coverage:")

    for rule, values in diagnostics["rule_coverage"].items():
        print(
            f"  {rule}: {values['count']:,} "
            f"({values['percentage']:.2f}%)  "
            f"only_this_function={values['only_this_function']:,}  "
            f"share_of_spam_labels={values['share_of_spam_labels']:.1f}%"
        )

    print(
        "\nReviews with multiple triggered rules: "
        f"{diagnostics['multiple_rule_reviews']:,} "
        f"({diagnostics['multiple_rule_percentage']:.2f}%)"
    )

    print("\nRule co-occurrence:")

    if diagnostics["rule_cooccurrence"]:
        for pair, count in diagnostics["rule_cooccurrence"].items():
            print(f"  {pair}: {count:,}")
    else:
        print("  No co-occurrence found.")

    print("=" * 60)
