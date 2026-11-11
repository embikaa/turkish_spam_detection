"""main.py — Turkish spam detection: single orchestration entry point.

Design rules enforced here:
  * one split, produced by one function, reused by every other script;
  * every transformer (TF-IDF, PCA, scaler) is fitted on the training split only;
  * model selection uses the validation split, reporting uses the test split;
  * every number that appears in the paper is written to results/metrics.json;
  * the rule-based labeling function is also evaluated as a classifier, so the
    circularity ceiling (Bulgu 1/2) is reported as a number instead of argued.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import sys
import warnings
from datetime import datetime

import numpy as np

try:
    from imblearn.over_sampling import RandomOverSampler

    HAS_IMBLEARN = True
except ImportError:  # imbalanced-learn is optional; see resample_training_set
    RandomOverSampler = None
    HAS_IMBLEARN = False

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import Config
from features import combine_features, create_three_way_features
from labeling import (
    get_labeling_diagnostics,
    label_texts,
    labeling_function,
    print_labeling_diagnostics,
)
from train import (
    evaluate_model,
    print_results,
    print_test_results,
    train_and_evaluate,
)
from visualize import create_all_plots

warnings.filterwarnings("ignore")


# ====
# 1. REPRODUCIBILITY
# ====

def set_seed(seed: int) -> None:
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


# ====
# 2. DATA LOADING
# ====

def load_data(
    data_path: str,
    sample_size: int | None = None,
    seed: int = 42,
) -> list:
    """Read the CSV as one review per line, header dropped."""
    print("Loading data...")

    try:
        with open(data_path, "r", encoding="utf-8-sig") as file:
            lines = file.read().splitlines()

    except FileNotFoundError:
        print(f"Error: file not found -> {data_path}")
        sys.exit(1)

    texts = [line.strip() for line in lines[1:] if line.strip()]

    if not texts:
        raise ValueError("No review texts were loaded.")

    if sample_size and sample_size < len(texts):
        rng = random.Random(seed)
        texts = rng.sample(texts, sample_size)
        print(f"Sampling mode: {sample_size:,} samples used.")

    print(f"Total loaded texts: {len(texts):,}")
    return texts


# ====
# 3. OPTIONAL DEDUPLICATION
# ====

def maybe_deduplicate(texts: list) -> tuple[list, dict | None]:
    """Remove exact duplicates before labeling and splitting.

    Duplicates must be removed before the split: otherwise the same review can
    land in train and test, which is a leakage source independent of the ANN
    early-stopping issue. Controlled by Config.DEDUP; silently skipped when the
    flag is off or dedup.py is absent, so this file stays runnable on its own.
    """
    if not getattr(Config, "DEDUP", False):
        return texts, None

    try:
        from dedup import dedupe_texts
    except ImportError:
        print("  [warn] Config.DEDUP is True but dedup.py was not found; skipping.")
        return texts, None

    unique_texts, _, stats, _ = dedupe_texts(texts)

    print(
        f"  Deduplication: {stats['n_total']:,} -> {stats['n_unique']:,} "
        f"({stats['duplicate_rate_pct']:.2f}% duplicates removed)"
    )

    return unique_texts, stats


# ====
# 4. SPLITTING
# ====

def split_data(
    texts: list,
    labels: np.ndarray,
    seed: int,
) -> tuple[list, list, list, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/validation/test split, produced in exactly one place.

    With TEST_SIZE=0.2 and VAL_SIZE=0.1 the proportions are 72% / 8% / 20%.
    The test split is drawn first so that it stays identical to earlier runs.
    """
    val_size = getattr(Config, "VAL_SIZE", 0.1)

    train_val_texts, test_texts, y_train_val, y_test = train_test_split(
        texts,
        labels,
        test_size=Config.TEST_SIZE,
        stratify=labels,
        random_state=seed,
    )

    train_texts, val_texts, y_train, y_val = train_test_split(
        train_val_texts,
        y_train_val,
        test_size=val_size,
        stratify=y_train_val,
        random_state=seed,
    )

    for name, y in (("train", y_train), ("validation", y_val), ("test", y_test)):
        counts = np.bincount(y, minlength=2)
        if counts.min() < 2:
            raise ValueError(
                f"The {name} split contains fewer than two observations of a class "
                f"({counts.tolist()}). Stratified splitting is not feasible."
            )

    return train_texts, val_texts, test_texts, y_train, y_val, y_test


# ====
# 5. LABEL STATISTICS COMPATIBILITY
# ====

def prepare_visualization_label_stats(diagnostics: dict) -> dict:
    """Map labeling diagnostics onto the keys visualize.py expects."""
    class_distribution = diagnostics["class_distribution"]
    class_percentages = diagnostics["class_percentages"]

    return {
        "total": diagnostics["total_reviews"],
        "genuine": class_distribution["genuine"],
        "spam": class_distribution["spam"],
        "genuine_pct": class_percentages["genuine"],
        "spam_pct": class_percentages["spam"],
        "rule_coverage": diagnostics["rule_coverage"],
        "rule_cooccurrence": diagnostics["rule_cooccurrence"],
        "multiple_rule_reviews": diagnostics["multiple_rule_reviews"],
        "multiple_rule_percentage": diagnostics["multiple_rule_percentage"],
        "diagnostics": diagnostics,
    }


# ====
# 5b. BALANCING (imbalanced-learn optional)
# ====

def resample_training_set(
    X: np.ndarray,
    y: np.ndarray,
    ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class in the training split only.

    Uses imbalanced-learn when installed and falls back to an equivalent
    numpy implementation otherwise, so a missing optional dependency cannot
    stop the pipeline. Both paths apply the same rule: the minority class is
    sampled with replacement up to `ratio` times the majority count.
    """
    y = np.asarray(y).astype(int).ravel()
    classes, counts = np.unique(y, return_counts=True)

    if len(classes) < 2:
        return X, y

    if HAS_IMBLEARN:
        sampler = RandomOverSampler(sampling_strategy=ratio, random_state=seed)
        return sampler.fit_resample(X, y)

    minority = int(classes[np.argmin(counts)])
    n_majority = int(counts.max())
    n_minority = int(counts.min())
    target = int(round(ratio * n_majority))

    if target <= n_minority:
        return X, y

    rng = np.random.default_rng(seed)
    minority_idx = np.flatnonzero(y == minority)
    extra_idx = rng.choice(minority_idx, size=target - n_minority, replace=True)
    order = np.concatenate([np.arange(len(y)), extra_idx])
    rng.shuffle(order)

    return X[order], y[order]


# ====
# 6. BASELINES
# ====

def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Always predict the majority class."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)

    y_pred = dummy.predict(np.zeros((len(y_test), 1)))
    return evaluate_model(y_test, y_pred)


def rule_baseline(test_texts: list, y_test: np.ndarray) -> dict:
    """Apply the labeling function directly to the test texts.

    This is the circularity ceiling (Bulgu 1/2). Because the weak labels were
    produced by the same function with the same defaults, the score is ~1.0 by
    construction. That is the point: it states, as a number, that the task the
    models are trained on is rule imitation, and it gives the upper bound that
    every learned model is compared against.
    """
    y_pred = np.asarray(
        [labeling_function(text)[0] for text in test_texts],
        dtype=np.int64,
    )
    return evaluate_model(y_test, y_pred)


# ====
# 7. METRICS EXPORT
# ====

def _json_safe(obj):
    """Convert numpy types and large arrays into JSON-serialisable values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        if obj.size <= 4096:
            return [round(float(v), 6) for v in obj.ravel()]
        return f"<​array shape={obj.shape}>"

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return round(float(obj), 6)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, float):
        return round(obj, 6)

    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj

    return str(obj)


def save_metrics(payload: dict, output_dir: str) -> str:
    """Write every reported number to one file (Bölüm 4 single source)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")

    with open(path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, ensure_ascii=False, indent=2)

    print(f"  Metrics written to: {path}")
    return path


def config_snapshot() -> dict:
    """Record the parameters that produced these numbers."""
    keys = (
        "BERT_MODEL", "MAX_LENGTH", "BATCH_SIZE", "USE_FP16",
        "TFIDF_FEATURES", "PCA_COMPONENTS", "TEST_SIZE", "VAL_SIZE",
        "SEED", "OVERSAMPLING_RATIO", "SAMPLE_SIZE", "DEDUP",
        "ANN_SELECTION_METRIC",
    )
    return {key: getattr(Config, key, None) for key in keys}


# ====
# 8. PLOT CALL COMPATIBILITY
# ====

def call_create_all_plots(**kwargs) -> None:
    """Pass only the arguments the installed visualize.py actually accepts.

    visualize.py gains a `split_label` parameter in its own fix round; until
    then the extra keyword is dropped here instead of raising TypeError.
    """
    supported = inspect.signature(create_all_plots).parameters
    create_all_plots(**{k: v for k, v in kwargs.items() if k in supported})


# ====
# 9. MAIN PIPELINE
# ====

def main(seed: int | None = None) -> None:
    seed = Config.SEED if seed is None else seed

    print("\n" + "=" * 60)
    print(" TURKISH SPAM DETECTION SYSTEM")
    print("=" * 60)
    print(f"Seed: {seed}")

    set_seed(seed)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # ----
    # Data loading and optional deduplication
    # ----

    texts = load_data(
        Config.DATA_PATH,
        sample_size=Config.SAMPLE_SIZE,
        seed=seed,
    )

    texts, dedup_stats = maybe_deduplicate(texts)

    # ----
    # Weak supervision / labeling
    # ----

    print("\n" + "-" * 60)
    print(" WEAK SUPERVISION AND LABELING")
    print("-" * 60)

    labels, triggered_rules = label_texts(texts)

    diagnostics = get_labeling_diagnostics(
        labels=labels,
        all_triggered_rules=triggered_rules,
    )
    print_labeling_diagnostics(diagnostics)

    label_stats = prepare_visualization_label_stats(diagnostics)

    del triggered_rules
    gc.collect()

    unique_classes, class_counts = np.unique(labels, return_counts=True)

    if len(unique_classes) < 2:
        raise ValueError(
            "The labeling process generated only one class. "
            "A binary classification experiment cannot proceed."
        )

    if np.min(class_counts) < 2:
        raise ValueError(
            "At least two observations per class are required "
            "for stratified splitting."
        )

    # ----
    # Train / Validation / Test split
    # ----

    print("\nSplitting dataset (Train/Validation/Test)...")

    (
        train_texts,
        val_texts,
        test_texts,
        y_train,
        y_val,
        y_test,
    ) = split_data(texts, labels, seed)

    del texts, labels
    gc.collect()

    print(f"Training set:   {len(y_train):,}")
    print(f"Validation set: {len(y_val):,}")
    print(f"Test set:       {len(y_test):,}")

    # ----
    # Feature extraction
    # ----

    print("\n" + "-" * 60)
    print(" FEATURE EXTRACTION")
    print("-" * 60)

    (
        X_train_tfidf,
        X_val_tfidf,
        X_test_tfidf,
        X_train_pca,
        X_val_pca,
        X_test_pca,
        tfidf_model,
        pca_model,
    ) = create_three_way_features(
        train_texts,
        val_texts,
        test_texts,
        max_features=Config.TFIDF_FEATURES,
        model_name=Config.BERT_MODEL,
        batch_size=Config.BATCH_SIZE,
        max_length=Config.MAX_LENGTH,
        use_fp16=Config.USE_FP16,
        n_components=Config.PCA_COMPONENTS,
        random_state=seed,
    )

    X_train = combine_features(X_train_tfidf, X_train_pca)
    X_val = combine_features(X_val_tfidf, X_val_pca)
    X_test = combine_features(X_test_tfidf, X_test_pca)

    del (
        X_train_tfidf, X_val_tfidf, X_test_tfidf,
        X_train_pca, X_val_pca, X_test_pca,
    )
    gc.collect()

    # ----
    # Scaling and balancing
    # ----

    print("\nPreparing data (Scaling & Balancing)...")

    # Fit on the training split only; cast to float32 to keep peak memory
    # near 0.9 GB instead of 1.8 GB for the resampled matrix.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    del X_train, X_val, X_test
    gc.collect()

    X_train_res, y_train_res = resample_training_set(
        X_train_scaled,
        y_train,
        ratio=Config.OVERSAMPLING_RATIO,
        seed=seed,
    )
    X_train_res = np.asarray(X_train_res, dtype=np.float32)

    print(
        f"Oversampling: {len(y_train):,} -> {len(y_train_res):,} "
        f"({'imbalanced-learn' if HAS_IMBLEARN else 'numpy fallback'})"
    )

    del X_train_scaled
    gc.collect()

    # ----
    # Model training
    # ----

    print("\n" + "-" * 60)
    print(" MODEL TRAINING")
    print("-" * 60)

    validation_results, test_results = train_and_evaluate(
        X_train_res,
        X_val_scaled,
        X_test_scaled,
        y_train_res,
        y_val,
        y_test,
        seed=seed,
    )

    print("\nVALIDATION RESULTS — used for model selection")
    print_results(validation_results)

    print_test_results(test_results, validation_results)

    # ----
    # Model selection: one metric, one place
    # ----

    ranked = [
        (name, metrics)
        for name, metrics in validation_results.items()
        if "error" not in metrics
    ]

    if not ranked:
        raise RuntimeError("Every model failed on the validation split.")

    best_name, best_val = max(ranked, key=lambda item: item[1]["macro_f1"])
    best_test = test_results[best_name]

    print(
        f"\nSELECTED MODEL (validation macro-F1): {best_name}  "
        f"val macro-F1={best_val['macro_f1']:.4f}  "
        f"test macro-F1={best_test['macro_f1']:.4f}  "
        f"test spam-F1={best_test['f1']:.4f}"
    )

    # ----
    # Baselines
    # ----

    print("\nBASELINES — TEST")

    base = majority_baseline(y_train, y_test)
    print(
        f"  Majority class:  Acc={base['accuracy']:.4f}  SpamF1={base['f1']:.4f}  "
        f"MacroF1={base['macro_f1']:.4f}  BalAcc={base['balanced_accuracy']:.4f}  "
        f"Genuine recall={base['genuine_recall']:.4f}"
    )

    rules = rule_baseline(test_texts, y_test)
    print(
        f"  Labeling rules:  Acc={rules['accuracy']:.4f}  SpamF1={rules['f1']:.4f}  "
        f"MacroF1={rules['macro_f1']:.4f}  BalAcc={rules['balanced_accuracy']:.4f}  "
        f"Spam precision={rules['precision']:.4f}  Spam recall={rules['recall']:.4f}"
    )
    print(
        "  (The rule baseline is ~1.0 by construction: it is the circularity "
        "ceiling, not a competing model.)"
    )

    # ----
    # Visualization and analysis
    # ----

    print("\n" + "-" * 60)
    print(" VISUALIZATION AND ANALYSIS")
    print("-" * 60)

    call_create_all_plots(
        results=test_results,
        split_label="Test",
        label_stats=label_stats,
        output_dir=Config.RESULTS_DIR,
        texts=train_texts + val_texts + test_texts,
        labels=np.concatenate([y_train, y_val, y_test]),
        X_embed=X_val_scaled,
        y_embed=y_val,
        embed_split_label="Validation",
        y_train_orig=y_train,
        y_train_res=y_train_res,
        pca_model=pca_model,
        tfidf_vectorizer=tfidf_model,
        seed=seed,
    )

    # ----
    # Metrics export
    # ----

    print("\n" + "-" * 60)
    print(" METRICS EXPORT")
    print("-" * 60)

    save_metrics(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "config": config_snapshot(),
            "labeling": diagnostics,
            "dedup": dedup_stats,
            "splits": {
                "train": int(len(y_train)),
                "validation": int(len(y_val)),
                "test": int(len(y_test)),
                "train_spam": int(y_train.sum()),
                "validation_spam": int(y_val.sum()),
                "test_spam": int(y_test.sum()),
            },
            "ranking_metric": "macro_f1",
            "selected_model": best_name,
            "validation": validation_results,
            "test": test_results,
            "baselines": {"majority": base, "labeling_rules": rules},
        },
        Config.RESULTS_DIR,
    )

    # ----
    # Completion
    # ----

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Outputs saved to: '{Config.RESULTS_DIR}'")
    print("=" * 60 + "\n")


# ====
# 10. ENTRY POINT
# ====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turkish spam detection pipeline")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override Config.SEED (used by run_seeds.py for multi-seed runs).",
    )
    args = parser.parse_args()

    main(seed=args.seed)
