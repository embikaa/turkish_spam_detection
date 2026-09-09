"""SHAP-based feature importance analysis for the persisted best model.

Loads the artifacts saved by `save_model.py` (TF-IDF vectorizer, PCA model,
scaler, best model) and computes SHAP values on a sample of the test set to
produce `results/10_shap_top15.png`.

The best model (ANN / MLPClassifier) is not tree-based, so `shap.TreeExplainer`
is not available. `shap.KernelExplainer` is used instead, which is
model-agnostic but computationally expensive — hence the background and
explanation sets are kept small (see `Config.SHAP_BACKGROUND_SIZE` /
`Config.SHAP_EXPLAIN_SIZE`).

Usage:
    python explainability.py
"""
import gc
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split

from config import Config
from labeling import label_texts, print_label_stats
from preprocessing import clean_texts

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(Config.BASE_DIR, "models")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_data(path: str, sample_size=None, seed: int = 42) -> list:
    print("Loading data...")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: File not found -> {path}")
        sys.exit(1)

    texts = [line for line in lines[1:] if line.strip()]
    if sample_size and sample_size < len(texts):
        import random
        rng = random.Random(seed)
        texts = rng.sample(texts, sample_size)
        print(f"Sampling mode: {sample_size:,} samples used.")

    print(f"Total loaded texts: {len(texts):,}")
    return texts


def load_artifacts():
    print("Loading persisted artifacts from 'models/'...")
    required = [
        "tfidf_vectorizer.joblib",
        "pca_model.joblib",
        "scaler.joblib",
        "best_model.joblib",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        print(f"Error: missing artifact(s) {missing}. Run 'python save_model.py' first.")
        sys.exit(1)

    tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    pca_model = joblib.load(os.path.join(MODELS_DIR, "pca_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    return tfidf_vectorizer, pca_model, scaler, best_model


def build_feature_names(tfidf_vectorizer, n_pca_components: int) -> list:
    """Feature order must match features.combine_features: TF-IDF, then BERT-PCA."""
    tfidf_names = list(tfidf_vectorizer.get_feature_names_out())
    pca_names = [f"bert_pca_{i}" for i in range(n_pca_components)]
    return tfidf_names + pca_names


def main():
    print("\n" + "=" * 60)
    print(" SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    set_seed(Config.SEED)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    tfidf_vectorizer, pca_model, scaler, best_model = load_artifacts()

    # Reconstruct the same test split used during training
    texts = load_data(Config.DATA_PATH, sample_size=Config.SAMPLE_SIZE, seed=Config.SEED)
    labels, label_stats = label_texts(texts, threshold=Config.SPAM_THRESHOLD)
    print_label_stats(label_stats)

    _, test_texts, _, y_test = train_test_split(
        texts, labels,
        test_size=Config.TEST_SIZE,
        stratify=labels,
        random_state=Config.SEED,
    )
    del texts
    gc.collect()

    # Keep a manageable subset: SHAP (KernelExplainer) scales poorly with
    # dataset size, so we only explain a sample of the test set.
    explain_size = min(Config.SHAP_EXPLAIN_SIZE, len(test_texts))
    import random
    rng = random.Random(Config.SEED)
    sample_idx = rng.sample(range(len(test_texts)), explain_size)
    sample_texts = [test_texts[i] for i in sample_idx]

    print(f"\nTransforming {explain_size} sampled test reviews into model features...")
    tfidf_clean = clean_texts(sample_texts, method="tfidf")
    bert_clean = clean_texts(sample_texts, method="bert")

    from features import create_bert_features  # local import: heavy (loads BERT)

    X_tfidf = tfidf_vectorizer.transform(tfidf_clean).toarray()

    # Re-embed the sampled texts with the same BERT model used in training,
    # then project with the already-fitted PCA model (no re-fitting).
    X_bert, _ = create_bert_features(
        bert_clean, bert_clean[:1],  # dummy second arg, only train output is used
        model_name=Config.BERT_MODEL,
        batch_size=Config.BATCH_SIZE,
        max_length=Config.MAX_LENGTH,
        use_fp16=Config.USE_FP16,
    )
    X_bert_pca = pca_model.transform(X_bert)

    X_combined = np.hstack([X_tfidf, X_bert_pca])
    X_scaled = scaler.transform(X_combined)
    del X_tfidf, X_bert, X_bert_pca, X_combined
    gc.collect()

    feature_names = build_feature_names(tfidf_vectorizer, pca_model.n_components_)

    # Background set for KernelExplainer: a small representative subset,
    # summarized with k-means to keep runtime manageable.
    bg_size = min(Config.SHAP_BACKGROUND_SIZE, X_scaled.shape[0])
    background = shap.kmeans(X_scaled, bg_size)

    print(f"\nRunning KernelExplainer (background={bg_size}, explained={explain_size})...")
    print("This step is the slowest part of the script — it may take a while.")

    def predict_spam_proba(x):
        return best_model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_spam_proba, background)
    shap_values = explainer.shap_values(X_scaled, nsamples="auto")

    # Rank features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_n = Config.SHAP_TOP_N
    top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_values = mean_abs_shap[top_idx]

    print(f"\nTop {top_n} features by mean |SHAP value|:")
    for name, val in zip(top_features, top_values):
        print(f"  {name:<30} {val:.4f}")

    print("\nGenerating plot...")
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_values[::-1], color="#4C72B0")
    plt.yticks(y_pos, top_features[::-1])
    plt.xlabel("Mean |SHAP value| (impact on spam probability)")
    plt.title(f"Top {top_n} Features — SHAP Feature Importance (ANN)")
    plt.tight_layout()

    output_path = os.path.join(Config.RESULTS_DIR, "10_shap_top15.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
