"""preprocessing.py — text cleaning for the TF-IDF and BERT branches.

Referee findings handled in this module:
  * Bulgu 6: the report asks for the exact behaviour of Stem(w) to be
    formalised. turkish_stem documents the algorithm precisely; the suffix
    list and its order are FROZEN because changing them would invalidate
    every number the pipeline has already produced. The report's
    comparative stemmer ablation (e.g. against Zemberek) is a separate
    experiment, not an edit here;
  * Bulgu 5 / §3.4: cleaning is stateless and per-text, so it cannot leak
    holdout information. The train-only fitted components (TF-IDF, PCA,
    scaler) live in features.py and main.py;
  * Bulgu 1 (control experiment): the TF-IDF branch strips punctuation,
    case and emoji-bearing tokens — i.e. it is already a representation in
    which the input signals of several labeling rules are masked. The
    report's control experiment compares it against the BERT branch, which
    keeps those signals.
"""
import re
from typing import List

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

NLTK_STOPWORDS = set(stopwords.words('turkish'))

EXTRA_STOPWORDS = {
    "bir", "bu", "şu", "o", "ne", "ve", "veya", "ama", "fakat", "çok", "gibi", "kadar",
    "daha", "olan", "olarak", "için", "ile", "de", "da", "mi", "mu", "mü", "mı", "en",
    "sonra", "önce", "böyle", "şöyle", "hiç", "her", "şey", "diye", "ise", "zaman",
    "kendi", "bile", "sadece", "artık", "tam", "göre", "tüm", "başka", "aynı", "gerçekten",
    "pek", "var", "yok", "isim", "ben", "sen", "biz", "siz", "onlar", "ya", "ki"
}

STOPWORDS = NLTK_STOPWORDS.union(EXTRA_STOPWORDS)

STEM_EXCEPTIONS = {
    "ürün", "iade", "kargo", "güzel", "aldım", "aldı", "orijinal",
    "tavsiye", "teslim", "sipariş", "kalite", "kaliteli", "satıcı",
    "kullanışlı", "harika", "mükemmel", "teşekkür"
}

# Order matters: the first suffix that matches (longest-first by intent)
# is stripped. Duplicates are removed on the first occurrence, so the
# effective order is identical to the historical list.
_SUFFIXES_WITH_DUPLICATES = [
    'lerimizden', 'larımızdan', 'lerinden', 'lardan',
    'lerimize', 'larımıza', 'lerine', 'larına',
    'lerimizde', 'larımızda', 'lerinde', 'larında',
    'lerimizin', 'larımızın', 'lerinin', 'larının',
    'lerimden', 'larımdan', 'lerinden', 'lerden',
    'leriyle', 'larıyla', 'lerimiz', 'larımız',
    'lerince', 'larınca', 'leştir', 'laştır',
    'leşme', 'laşma', 'sinden', 'sından',
    'sine', 'sına', 'lerime', 'larıma',
    'lerimde', 'larımda', 'lerimle', 'larımla',
    'lerden', 'lerden', 'lerde', 'larda',
    'lerin', 'ların', 'lere', 'lara',
    'lerim', 'larım', 'leri', 'ları',
    'imiz', 'ımız', 'sizde', 'sında',
    'ler', 'lar',
    'imiz', 'ımız', 'siniz', 'sınız',
    'im', 'ım', 'üm', 'um',
    'in', 'ın', 'ün', 'un',
    'si', 'sı', 'sü', 'su',
    'sin', 'sın', 'sün', 'sun',
    'miz', 'mız', 'müz', 'muz',
    'niz', 'nız', 'nüz', 'nuz',
    'den', 'dan', 'ten', 'tan',
    'de', 'da', 'te', 'ta',
    'e', 'a',
    'iyor', 'ıyor', 'üyor', 'uyor',
    'ecek', 'acak',
    'miş', 'mış', 'müş', 'muş',
    'di', 'dı', 'dü', 'du',
    'ti', 'tı', 'tü', 'tu',
    'me', 'ma',
    'mek', 'mak',
    'dim', 'dım', 'düm', 'dum',
    'tim', 'tım', 'tüm', 'tum',
    'cık', 'cik', 'cuk', 'cük',
    'çık', 'çik', 'çuk', 'çük',
]

SUFFIXES = list(dict.fromkeys(_SUFFIXES_WITH_DUPLICATES))

RE_URL = re.compile(r"http\S+|www.\S+")
RE_PUNCTUATION = re.compile(r"[^\w\s]")
RE_WHITESPACE = re.compile(r"\s+")


def turkish_stem(word: str) -> str:
    """Stem(w) exactly as Bulgu 6 asks it to be formalised.

    Algorithm (single pass, no recursion, no morphological analysis):
      1. if len(w) <= 2 or w is in STEM_EXCEPTIONS, return w unchanged;
      2. scan SUFFIXES in list order and strip the FIRST suffix s such
         that w ends with s and the remainder keeps at least 3 characters;
      3. if no suffix applies, return w unchanged.
    List order, not longest match, decides ties; at most one suffix is
    removed; there is no vowel-harmony or morpheme-boundary check. The
    list and its order are frozen: any change alters every TF-IDF vector
    and therefore every reported number.
    """
    if len(word) <= 2 or word in STEM_EXCEPTIONS:
        return word

    for suffix in SUFFIXES:
        if len(word) > len(suffix) + 1 and word.endswith(suffix):
            stemmed = word[:-len(suffix)]
            if len(stemmed) >= 3:
                return stemmed
    return word


def clean_for_bert(text: str) -> str:
    """Whitespace normalisation only.

    Case, punctuation, emoji and URLs are kept: the cased Turkish BERT
    (Config.BERT_MODEL) was pretrained on raw text, and keeping the rules'
    input signals in this branch is what makes the Bulgu 1 control
    experiment meaningful (masked TF-IDF branch vs unmasked BERT branch).
    """
    if not isinstance(text, str):
        return ""
    return RE_WHITESPACE.sub(" ", text).strip()


def clean_for_tfidf(text: str) -> str:
    """Lowercase, URL strip, punctuation strip, stopword filter, Turkish stemming."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = RE_URL.sub("", text)
    text = RE_PUNCTUATION.sub(" ", text)
    text = RE_WHITESPACE.sub(" ", text).strip()

    words = text.split()
    processed_words = [
        turkish_stem(w) for w in words
        if w not in STOPWORDS and len(w) > 2
    ]

    return " ".join(processed_words)


def clean_texts(texts: List[str], method: str = "tfidf") -> List[str]:
    """Apply the branch-specific cleaner ('tfidf' or 'bert') to every text."""
    if method == "bert":
        return [clean_for_bert(t) for t in texts]
    return [clean_for_tfidf(t) for t in texts]
