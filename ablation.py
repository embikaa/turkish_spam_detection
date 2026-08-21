import gc
import random
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import Config
from preprocessing import clean_texts
from labeling import label_texts
from features import create_tfidf_features, create_bert_features, apply_pca, combine_features

BERT_MODEL   = Config.BERT_MODEL
TFIDF_FEATS  = Config.TFIDF_FEATURES
PCA_COMPS    = Config.PCA_COMPONENTS
MAX_LENGTH   = Config.MAX_LENGTH
BATCH_SIZE   = Config.BATCH_SIZE
USE_FP16     = Config.USE_FP16
TEST_SIZE    = Config.TEST_SIZE
SEED         = Config.SEED

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def train_eval(X_tr, X_te, y_tr, y_te, label):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    ros = RandomOverSampler(sampling_strategy=1.0, random_state=SEED)
    X_res, y_res = ros.fit_resample(X_tr_s, y_tr)
    ann = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        solver="adam", alpha=0.001,
        learning_rate="adaptive", max_iter=300,
        early_stopping=True, batch_size=512,
        random_state=SEED,
    )
    ann.fit(X_res, y_res)
    y_pred = ann.predict(X_te_s)
    f1  = f1_score(y_te, y_pred)
    acc = accuracy_score(y_te, y_pred)
    print(f"{label}: Accuracy={acc:.4f}, F1={f1:.4f}")
    return f1, acc

# --- Veri ---
set_seed(SEED)
print("Veri yükleniyor...")
with open(Config.DATA_PATH, 'r', encoding='utf-8-sig') as f:
    lines = f.read().splitlines()
texts = [l for l in lines[1:] if l.strip()]
print(f"Toplam: {len(texts):,}")

labels, stats = label_texts(texts, threshold=Config.SPAM_THRESHOLD)
print(f"Spam: {stats['spam_pct']}%  Genuine: {stats['genuine_pct']}%")

train_texts, test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=SEED
)
del texts; gc.collect()
print(f"Train: {len(train_texts):,}  Test: {len(test_texts):,}")

# --- Ön işleme ---
print("\nÖn işleme...")
train_tfidf_c = clean_texts(train_texts, method="tfidf")
test_tfidf_c  = clean_texts(test_texts,  method="tfidf")
train_bert_c  = clean_texts(train_texts, method="bert")
test_bert_c   = clean_texts(test_texts,  method="bert")

# --- TF-IDF ---
print("\nTF-IDF çıkarılıyor...")
X_tr_tfidf, X_te_tfidf, _ = create_tfidf_features(
    train_tfidf_c, test_tfidf_c, max_features=TFIDF_FEATS
)
del train_tfidf_c, test_tfidf_c; gc.collect()

# --- BERT ---
print("\nBERT çıkarılıyor (FP16)...")
X_tr_bert, X_te_bert = create_bert_features(
    train_bert_c, test_bert_c,
    model_name=BERT_MODEL,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    use_fp16=USE_FP16,
)
del train_bert_c, test_bert_c; gc.collect()

# --- PCA ---
print("\nPCA uygulanıyor...")
X_tr_bert_pca, X_te_bert_pca, _ = apply_pca(
    X_tr_bert, X_te_bert, n_components=PCA_COMPS, random_state=SEED
)
del X_tr_bert, X_te_bert; gc.collect()

# --- Ablasyon ---
print("\n" + "="*50)
print(" ABLATION STUDY")
print("="*50)

print("\n[1/3] TF-IDF Only...")
train_eval(X_tr_tfidf, X_te_tfidf, y_train, y_test, "TF-IDF Only")

print("\n[2/3] BERT Only...")
train_eval(X_tr_bert_pca, X_te_bert_pca, y_train, y_test, "BERT Only")

print("\n[3/3] TF-IDF + BERT (Hybrid)...")
X_tr_h = combine_features(X_tr_tfidf, X_tr_bert_pca)
X_te_h = combine_features(X_te_tfidf, X_te_bert_pca)
train_eval(X_tr_h, X_te_h, y_train, y_test, "TF-IDF + BERT (Hybrid)")
