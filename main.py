"""Main pipeline orchestration for Turkish spam detection system."""
import gc
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

from config import Config
from preprocessing import clean_texts
from labeling import label_texts, print_label_stats
from features import (
    create_tfidf_features,
    create_bert_features,
    apply_pca,
    combine_features,
)
from train import train_and_evaluate, print_results
from visualize import create_all_plots

warnings.filterwarnings("ignore")


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_data(data_path: str, sample_size: int = None, seed: int = 42) -> list:
    """
    Load and sample text data from CSV file.

    Args:
        data_path: Path to the CSV data file.
        sample_size: Optional number of samples to subsample.
        seed: Random seed for sampling.

    Returns:
        List of text strings from the dataset.
    """
    print("Loading data...")
    try:
        df = pd.read_csv(data_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File not found -> {data_path}")
        sys.exit(1)

    text_column = "comment" if "comment" in df.columns else df.columns[0]
    df = df.dropna(subset=[text_column])

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
        print(f"Sampling mode: {sample_size:,} samples used.")

    texts = df[text_column].tolist()
    print(f"Total loaded texts: {len(texts):,}")
    return texts


def main():
    """Execute the complete spam detection pipeline."""
    print("\n" + "=" * 60)
    print(" TURKISH SPAM DETECTION SYSTEM")
    print("=" * 60)

    set_seed(Config.SEED)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # Load data and apply weak labeling
    texts = load_data(Config.DATA_PATH, sample_size=Config.SAMPLE_SIZE, seed=Config.SEED)
    labels, label_stats = label_texts(texts, threshold=Config.SPAM_THRESHOLD)
    print_label_stats(label_stats)

    # Train/test split
    print("\nSplitting dataset (Train/Test)...")
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts, labels,
        test_size=Config.TEST_SIZE,
        stratify=labels,
        random_state=Config.SEED,
    )
    del texts
    gc.collect()

    print(f"Training set: {len(y_train):,}")
    print(f"Test set: {len(y_test):,}")

    # Text preprocessing
    print("\nPreprocessing texts...")
    print(" [1/2] Cleaning training data...")
    train_tfidf_clean = clean_texts(train_texts, method="tfidf")
    train_bert_clean = clean_texts(train_texts, method="bert")
    print(" [2/2] Cleaning test data...")
    test_tfidf_clean = clean_texts(test_texts, method="tfidf")
    test_bert_clean = clean_texts(test_texts, method="bert")

    # Feature extraction
    print("\n" + "-" * 60)
    print(" FEATURE EXTRACTION")
    print("-" * 60)

    X_train_tfidf, X_test_tfidf, tfidf_model = create_tfidf_features(
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

    # Data preparation (scaling and balancing)
    print("\nPreparing data (Scaling & Balancing)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ros = RandomOverSampler(
        sampling_strategy=Config.OVERSAMPLING_RATIO,
        random_state=Config.SEED,
    )
    X_train_res, y_train_res = ros.fit_resample(X_train_scaled, y_train)

    print("Scaling completed.")
    print(f"Oversampling: {len(y_train):,} -> {len(y_train_res):,} samples")
    del X_train, X_train_scaled
    gc.collect()

    # Model training and evaluation
    print("\n" + "-" * 60)
    print(" MODEL TRAINING")
    print("-" * 60)

    results = train_and_evaluate(
        X_train_res, X_test_scaled, y_train_res, y_test, seed=Config.SEED
    )
    print_results(results)

    # Visualization
    print("\n" + "-" * 60)
    print(" VISUALIZATION AND ANALYSIS")
    print("-" * 60)

    create_all_plots(
        results=results,
        label_stats=label_stats,
        output_dir=Config.RESULTS_DIR,
        texts=train_texts + test_texts,
        labels=np.concatenate([y_train, y_test]),
        X_train=X_train_res,
        X_test=X_test_scaled,
        y_train_orig=y_train,
        y_train_res=y_train_res,
        y_test=y_test,
        pca_model=pca_model,
        tfidf_vectorizer=tfidf_model,
        seed=Config.SEED,
    )

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Outputs saved to: '{Config.RESULTS_DIR}'")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
