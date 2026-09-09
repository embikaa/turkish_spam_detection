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

SUFFIXES = [
    'lerimizden', 'larımızdan', 'lerinden', 'lardan',
    'lerimize', 'larımıza', 'lerine', 'larına',
    'lerimizde', 'larımızda', 'lerinde', 'larında',
    'lerimizin', 'larımızın', 'lerinin', 'larının',
    'lerimden', 'larımdan', 'lerinden', 'lardan',
    'leriyle', 'larıyla', 'lerimiz', 'larımız',
    'lerince', 'larınca', 'leştir', 'laştır',
    'leşme', 'laşma', 'sinden', 'sından',
    'sine', 'sına', 'lerime', 'larıma',
    'lerimde', 'larımda', 'lerimle', 'larımla',
    'lerden', 'lardan', 'lerde', 'larda',
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

RE_URL = re.compile(r"http\S+|www.\S+")
RE_PUNCTUATION = re.compile(r"[^\w\s]")
RE_WHITESPACE = re.compile(r"\s+")

def turkish_stem(word: str) -> str:
    if len(word) <= 2 or word in STEM_EXCEPTIONS:
        return word

    for suffix in SUFFIXES:
        if len(word) > len(suffix) + 1 and word.endswith(suffix):
            stemmed = word[:-len(suffix)]
            if len(stemmed) >= 3:
                return stemmed
    return word

def clean_for_bert(text: str) -> str:
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
    if method == "bert":
        return [clean_for_bert(t) for t in texts]
    return [clean_for_tfidf(t) for t in texts]
