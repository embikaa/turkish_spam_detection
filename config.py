"""Configuration settings for the Turkish spam detection system.

Single source of truth for every hyperparameter that appears in the paper;
the hyperparameter table the report asks for (Bulgu 8) is read from here
and from train.get_models().
"""
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
    SAMPLE_SIZE = None

    # Multi-seed protocol (Bulgu 4): run_seeds.py repeats the full pipeline
    # with these seeds and reports mean ± std plus McNemar/bootstrap tests.
    SEEDS = [42, 0, 1, 2, 3]

    # Paired bootstrap resamples for the between-model significance tests.
    N_BOOTSTRAP = 1000

    # Legacy: the labeling API is threshold-free now; kept only so older
    # ad-hoc scripts do not crash. Not used by the pipeline any more.
    SPAM_THRESHOLD = 1.0

    # Explainability (SHAP) settings
    SHAP_BACKGROUND_SIZE = 100  # background samples used to approximate E[f(x)]
    SHAP_EXPLAIN_SIZE = 200     # test samples SHAP values are computed for
    SHAP_TOP_N = 15             # number of top features shown in the plot
