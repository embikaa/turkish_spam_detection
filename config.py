"""Configuration settings for the Turkish spam detection system."""
import os


class Config:
    """Central configuration for all pipeline parameters."""

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "veri_seti_200k.csv")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    # BERT model settings
    BERT_MODEL = "dbmdz/bert-base-turkish-cased"
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    USE_FP16 = True

    # Feature extraction settings
    TFIDF_FEATURES = 500
    PCA_COMPONENTS = 256

    # Training settings
    TEST_SIZE = 0.2
    SEED = 42
    OVERSAMPLING_RATIO = 1.0
    SAMPLE_SIZE = 20000
    SPAM_THRESHOLD = 0.8
