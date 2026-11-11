"""Feature extraction module for hybrid TF-IDF + BERT representation.

Referee findings handled in this module:
  * Bulgu 5 / §3.4: the TF-IDF vocabulary (min_df and max_df included) and
    the PCA projection are fitted on the training split only; validation and
    test are only transformed. Oversampling comes after these steps and
    touches the training split only (main.resample_training_set);
  * Bulgu 6 / §5.3: the TF-IDF settings are written out explicitly so the
    code matches the corrected formula: tf = raw term count, idf =
    ln((1+n)/(1+df)) + 1, document vectors L2-normalised row-wise;
  * §5.4: PCA is applied to the BERT embeddings only and is estimated from
    the training set; the hybrid space is TF-IDF + BERT-PCA;
  * Bulgu 8: the BERT embedding extraction cost (device, seconds, ms/sample)
    is measured and written to results/feature_timing.json.
"""
from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from config import Config
from preprocessing import STOPWORDS

FEATURE_TIMING_PATH = os.path.join(Config.RESULTS_DIR, "feature_timing.json")


def create_tfidf_features(
    train_texts: list,
    test_texts: list,
    max_features: int = 500,
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Fit TF-IDF on the training split; transform the holdout split.

    Bulgu 5 / §3.4: the vocabulary, the df statistics and the min_df/max_df
    filtering are estimated on the training split only — never on validation
    or test, which would leak holdout information into the feature space.
    Bulgu 6 / §5.3: the settings below are exactly what the corrected
    equation in the paper describes.
    """
    print("  TF-IDF extraction running...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(STOPWORDS),
        dtype=np.float32,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
        # Bulgu 6 / §5.3: made explicit so the code and the corrected
        # equation cannot drift apart. These are the sklearn defaults.
        sublinear_tf=False,  # tf is the raw term count
        smooth_idf=True,     # idf = ln((1+n)/(1+df)) + 1
        norm="l2",           # document vectors are L2-normalised row-wise
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
    use_fp16: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """CLS token embeddings from last hidden state (inference only).

    Bulgu 8: the caller measures and records the wall-clock cost of this
    extraction; the device is printed here.
    """
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
                # model.half() already runs the forward pass in fp16 on CUDA;
                # an autocast wrapper on top of it would be redundant.
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
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on the training split; project the holdout split.

    Bulgu 5 / §3.4 and §5.4: W_PCA is estimated from the training-set BERT
    embeddings only; validation and test are only projected. The randomized
    solver is seeded, so the projection is reproducible for a given seed.
    """
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
    """Concatenate the TF-IDF and BERT-PCA blocks (§5.4 layout)."""
    print("  Combining features...")
    X_combined = np.hstack([X_tfidf, X_bert])
    print(
        f"  ✓ V_hybrid = TF-IDF({X_tfidf.shape[1]}) + BERT-PCA({X_bert.shape[1]}) "
        f"= {X_combined.shape[1]} features (PCA applied to BERT only)"
    )
    return X_combined


def create_three_way_features(
    train_texts,
    val_texts,
    test_texts,
    max_features,
    model_name,
    batch_size,
    max_length,
    use_fp16,
    n_components,
    random_state,
):
    """Fit TF-IDF and PCA on train only; transform validation and test.

    §3.4 pipeline order: split → TF-IDF fit (train) → BERT extraction →
    PCA fit (train) → combination → scaler fit (train) → validation split →
    oversampling → training. This function covers the first four steps and
    returns the TF-IDF and BERT-PCA matrices separately so the ablation
    study can reuse them; scaling and oversampling stay with the callers.

    Bulgu 8: stage-wise extraction cost is written to
    results/feature_timing.json.
    """
    from preprocessing import clean_texts

    n_val = len(val_texts)
    holdout_texts = val_texts + test_texts

    train_tfidf_clean = clean_texts(train_texts, method="tfidf")
    holdout_tfidf_clean = clean_texts(holdout_texts, method="tfidf")

    t0 = time.perf_counter()
    X_train_tfidf, X_holdout_tfidf, tfidf_model = create_tfidf_features(
        train_tfidf_clean,
        holdout_tfidf_clean,
        max_features=max_features,
    )
    tfidf_seconds = time.perf_counter() - t0

    del train_tfidf_clean, holdout_tfidf_clean
    gc.collect()

    train_bert_clean = clean_texts(train_texts, method="bert")
    holdout_bert_clean = clean_texts(holdout_texts, method="bert")

    t0 = time.perf_counter()
    X_train_bert, X_holdout_bert = create_bert_features(
        train_bert_clean,
        holdout_bert_clean,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        use_fp16=use_fp16,
    )
    bert_seconds = time.perf_counter() - t0

    del train_bert_clean, holdout_bert_clean
    gc.collect()

    t0 = time.perf_counter()
    X_train_pca, X_holdout_pca, pca_model = apply_pca(
        X_train_bert,
        X_holdout_bert,
        n_components=n_components,
        random_state=random_state,
    )
    pca_seconds = time.perf_counter() - t0

    del X_train_bert, X_holdout_bert
    gc.collect()

    n_total = len(train_texts) + n_val + len(test_texts)
    timing = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "bert_model": model_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "use_fp16": bool(use_fp16 and torch.cuda.is_available()),
        "n_train": len(train_texts),
        "n_validation": n_val,
        "n_test": len(test_texts),
        "tfidf_fit_transform_seconds": round(tfidf_seconds, 2),
        "bert_embedding_seconds": round(bert_seconds, 2),
        "bert_ms_per_sample": round(bert_seconds * 1000.0 / n_total, 3),
        "pca_fit_transform_seconds": round(pca_seconds, 2),
    }

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    with open(FEATURE_TIMING_PATH, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    print(
        f"  ✓ Extraction cost: TF-IDF {timing['tfidf_fit_transform_seconds']}s, "
        f"BERT {timing['bert_embedding_seconds']}s "
        f"({timing['bert_ms_per_sample']} ms/sample), "
        f"PCA {timing['pca_fit_transform_seconds']}s"
    )
    print(f"  ✓ Timings written to {FEATURE_TIMING_PATH}")

    return (
        X_train_tfidf,
        X_holdout_tfidf[:n_val],
        X_holdout_tfidf[n_val:],
        X_train_pca,
        X_holdout_pca[:n_val],
        X_holdout_pca[n_val:],
        tfidf_model,
        pca_model,
    )
