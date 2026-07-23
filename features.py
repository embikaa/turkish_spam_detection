"""Feature extraction module for hybrid TF-IDF + BERT representation."""
import gc
from typing import Tuple, Any

import numpy as np
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from preprocessing import STOPWORDS


def create_tfidf_features(
    train_texts: list,
    test_texts: list,
    max_features: int = 500
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Unigrams+bigrams, filters rare and overly common terms."""
    print("  TF-IDF extraction running...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(STOPWORDS),
        dtype=np.float32,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85
    )

    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    print(f"  ✓ TF-IDF dimension: {X_train.shape[1]}")
    return X_train, X_test, vectorizer


def create_bert_features(
    train_texts: list,
    test_texts: list,
    model_name: str,
    batch_size: int = 32,
    max_length: int = 128,
    use_fp16: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """CLS token embeddings from last hidden state."""
    from transformers import AutoTokenizer, AutoModel

    print(f"  Extracting BERT embeddings ({model_name})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    if use_fp16 and device.type == "cuda":
        model = model.half()

    def extract_embeddings(texts):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  Processing"):
            batch_texts = texts[i:i + batch_size]
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                if use_fp16 and device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        output = model(**tokens)
                else:
                    output = model(**tokens)

                cls_vectors = output.last_hidden_state[:, 0, :].float().cpu().numpy()
                embeddings.append(cls_vectors)

            del tokens, output
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return np.vstack(embeddings)

    X_train = extract_embeddings(train_texts)
    X_test = extract_embeddings(test_texts)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  ✓ BERT dimension: {X_train.shape[1]}")
    return X_train, X_test


def apply_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 256,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    print(f"  Applying PCA ({X_train.shape[1]} -> {n_components})...")

    pca = PCA(
        n_components=n_components,
        random_state=random_state,
        svd_solver='randomized'
    )

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"  Explained variance: {explained_var:.1f}%")

    return X_train_pca, X_test_pca, pca


def combine_features(X_tfidf: np.ndarray, X_bert: np.ndarray) -> np.ndarray:
    print("  Combining features...")
    X_combined = np.hstack([X_tfidf, X_bert])
    print(f"  ✓ Total feature count: {X_combined.shape[1]}")
    return X_combined
