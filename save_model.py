"""
Train the ANN pipeline and persist all artifacts to models/.

Run once:  python save_model.py
Outputs:
  models/tfidf_vectorizer.joblib
  models/pca_model.joblib
  models/scaler.joblib
  models/best_model.joblib   (ANN — MLPClassifier)
  models/model_info.json
"""
import gc
import json
import os
import sys
import joblib
import warnings
from datetime import datetime

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import Config
from features import apply_pca, combine_features, create_bert_features, create_tfidf_features
from labeling import label_texts, print_label_stats
from preprocessing import clean_texts
from train import evaluate_model

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(Config.BASE_DIR, "models")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_data(path: str, sample_size=None, seed: int = 42) -> list:
    print("Veri yükleniyor...")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Hata: Dosya bulunamadı -> {path}")
        sys.exit(1)

    texts = [line for line in lines[1:] if line.strip()]
    if sample_size and sample_size < len(texts):
        import random
        rng = random.Random(seed)
        texts = rng.sample(texts, sample_size)
        print(f"Örnekleme: {sample_size:,} adet kullanıldı.")

    print(f"Toplam metin: {len(texts):,}")
    return texts


def main():
    print("\n" + "=" * 60)
    print(" MODEL EĞİTİM VE KAYDETME")
    print("=" * 60)

    set_seed(Config.SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # --- Veri ---
    texts = load_data(Config.DATA_PATH, sample_size=Config.SAMPLE_SIZE, seed=Config.SEED)
    labels, label_stats = label_texts(texts, threshold=Config.SPAM_THRESHOLD)
    print_label_stats(label_stats)

    train_texts, test_texts, y_train, y_test = train_test_split(
        texts, labels,
        test_size=Config.TEST_SIZE,
        stratify=labels,
        random_state=Config.SEED,
    )
    del texts
    gc.collect()

    # --- Ön işleme ---
    print("\nMetinler ön işleniyor...")
    train_tfidf_clean = clean_texts(train_texts, method="tfidf")
    train_bert_clean = clean_texts(train_texts, method="bert")
    test_tfidf_clean = clean_texts(test_texts, method="tfidf")
    test_bert_clean = clean_texts(test_texts, method="bert")

    # --- Özellik çıkarımı ---
    print("\nÖzellikler çıkarılıyor...")
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = create_tfidf_features(
        train_tfidf_clean, test_tfidf_clean, max_features=Config.TFIDF_FEATURES
    )
    del train_tfidf_clean, test_tfidf_clean
    gc.collect()

    X_train_bert, X_test_bert = create_bert_features(
        train_bert_clean, test_bert_clean,
        model_name=Config.BERT_MODEL,
        batch_size=Config.BATCH_SIZE,
        max_length=Config.MAX_LENGTH,
        use_fp16=Config.USE_FP16,
    )
    del train_bert_clean, test_bert_clean
    gc.collect()

    X_train_bert_pca, X_test_bert_pca, pca_model = apply_pca(
        X_train_bert, X_test_bert,
        n_components=Config.PCA_COMPONENTS,
        random_state=Config.SEED,
    )
    del X_train_bert, X_test_bert
    gc.collect()

    X_train = combine_features(X_train_tfidf, X_train_bert_pca)
    X_test = combine_features(X_test_tfidf, X_test_bert_pca)
    del X_train_tfidf, X_test_tfidf, X_train_bert_pca, X_test_bert_pca
    gc.collect()

    # --- Ölçekleme & dengeleme ---
    print("\nVeri hazırlanıyor...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ros = RandomOverSampler(sampling_strategy=Config.OVERSAMPLING_RATIO, random_state=Config.SEED)
    X_train_res, y_train_res = ros.fit_resample(X_train_scaled, y_train)
    del X_train, X_train_scaled
    gc.collect()

    # --- Model eğitimi (ANN) ---
    print("\nANN modeli eğitiliyor...")
    ann = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        max_iter=300,
        early_stopping=True,
        batch_size=512,
        random_state=Config.SEED,
    )
    ann.fit(X_train_res, y_train_res)
    y_pred = ann.predict(X_test_scaled)
    y_proba = ann.predict_proba(X_test_scaled)[:, 1]
    best_metrics = evaluate_model(y_test, y_pred, y_proba)
    best_name = "ANN"
    best_model = ann
    print(f"  F1: {best_metrics['f1']:.4f}  AUC: {best_metrics.get('auc', 0):.4f}")

    # --- Kaydet ---
    print("\nArtifact'lar kaydediliyor...")
    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(pca_model,        os.path.join(MODELS_DIR, "pca_model.joblib"))
    joblib.dump(scaler,           os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(best_model,       os.path.join(MODELS_DIR, "best_model.joblib"))

    model_info = {
        "best_model_name": best_name,
        "bert_model": Config.BERT_MODEL,
        "tfidf_features": Config.TFIDF_FEATURES,
        "pca_components": Config.PCA_COMPONENTS,
        "max_length": Config.MAX_LENGTH,
        "metrics": {
            k: v for k, v in best_metrics.items()
            if isinstance(v, (int, float, str))
        },
        "trained_at": datetime.now().isoformat(),
        "label_stats": label_stats,
    }
    with open(os.path.join(MODELS_DIR, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f" Kaydedilen konum: {MODELS_DIR}/")
    print("  - tfidf_vectorizer.joblib")
    print("  - pca_model.joblib")
    print("  - scaler.joblib")
    print("  - best_model.joblib")
    print("  - model_info.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
