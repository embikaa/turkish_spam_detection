"""save_model.py — persist the model that main.py selected, with its frozen threshold.

Referee findings handled in this module:
  * Bulgu 4 / Section 4: no independent model choice. The model to persist is
    read from results/metrics.json (selected by main.py on validation macro-F1),
    and the persisted model's metrics are cross-checked against that file;
  * Bulgu 3: the ANN is early-stopped on the non-resampled validation split
    (train.fit_ann_with_validation); oversampling touches the training split only;
  * Bulgu 5 / Section 3.4: identical split and preprocessing, produced by the
    same functions main.py uses; TF-IDF, PCA and the scaler are fitted on the
    training split only;
  * Section 6 (threshold): the decision threshold is selected on validation
    with train.select_threshold (the same rule main.py uses), frozen for test,
    and stored with the model so downstream scripts reuse it.

Usage:
    python main.py --seed 42        # must run first; writes results/metrics.json
    python save_model.py --seed 42
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import Config
from features import combine_features, create_three_way_features
from labeling import label_texts
from main import (
    HAS_IMBLEARN,
    _json_safe,
    config_snapshot,
    load_data,
    maybe_deduplicate,
    resample_training_set,
    set_seed,
    split_data,
)
from train import (
    RANKING_METRIC,
    _predict_proba,
    evaluate_model,
    fit_ann_with_validation,
    get_models,
    predict_with_threshold,
    select_threshold,
)

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(Config.BASE_DIR, "models")
METRICS_PATH = os.path.join(Config.RESULTS_DIR, "metrics.json")

CONSISTENCY_TOL = 1e-3
CHECK_KEYS = (
    "threshold",
    "macro_f1",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "genuine_recall",
)


# ====
# 1. SELECTION FROM THE SINGLE SOURCE
# ====

def load_reference(seed: int) -> dict:
    """Read the metrics file written by main.py for the same seed."""
    if not os.path.exists(METRICS_PATH):
        print(f"Error: {METRICS_PATH} not found. Run 'python main.py --seed {seed}' first.")
        sys.exit(1)

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        reference = json.load(file)

    if reference.get("seed") != seed:
        print(
            f"Error: metrics.json was produced with seed={reference.get('seed')}, "
            f"but save_model.py was called with seed={seed}. "
            "Re-run main.py with the same seed."
        )
        sys.exit(1)

    if "selected_model" not in reference:
        print("Error: metrics.json has no 'selected_model' entry. Re-run main.py.")
        sys.exit(1)

    return reference


def check_splits(reference: dict, actual: dict) -> None:
    """Abort if the reconstructed split differs from the one main.py used."""
    expected = reference.get("splits", {})

    for key, value in actual.items():
        if key in expected and int(expected[key]) != int(value):
            print(
                f"Error: {key} split size {value:,} differs from metrics.json "
                f"({int(expected[key]):,}). Data, DEDUP or SAMPLE_SIZE changed "
                "since main.py was run."
            )
            sys.exit(1)


def check_against_reference(name: str, split: str, metrics: dict, reference: dict) -> bool:
    """Compare the persisted model's metrics with the numbers in metrics.json."""
    ref = reference.get(split, {}).get(name, {})

    if not ref:
        print(f"  [warn] No {split} entry for '{name}' in metrics.json; check skipped.")
        return False

    consistent = True
    print(f"\n  Consistency check ({split}) against metrics.json:")

    for key in CHECK_KEYS:
        if key not in ref or key not in metrics:
            continue
        if ref[key] is None or metrics[key] is None:
            continue

        diff = abs(float(metrics[key]) - float(ref[key]))
        ok = diff <= CONSISTENCY_TOL
        consistent &= ok

        print(
            f"    {key:<18} saved={float(metrics[key]):.4f}  "
            f"metrics.json={float(ref[key]):.4f}  {'ok' if ok else 'MISMATCH'}"
        )

    return consistent


# ====
# 2. TRAINING THE SELECTED MODEL
# ====

def train_selected(name, seed, X_train, y_train, X_val, y_val):
    """Train exactly the model main.py selected, with main.py's settings."""
    model = get_models(seed).get(name)

    if model is None:
        print(f"Error: model '{name}' is not available in train.get_models().")
        sys.exit(1)

    if name == "ANN":
        # Same early-stopping metric as train.train_and_evaluate (threshold_metric
        # defaults to RANKING_METRIC there); passed only if the function accepts it.
        kwargs = {}
        if "metric" in inspect.signature(fit_ann_with_validation).parameters:
            kwargs["metric"] = RANKING_METRIC
        # fit_ann_with_validation returns (best_model, history); this wrapper's
        # contract is "return the fitted model", so unwrap here. The history is
        # not consumed by save_model.py.
        model, ann_history = fit_ann_with_validation(
            model, X_train, y_train, X_val, y_val, **kwargs
        )
        return model

    model.fit(X_train, y_train)
    return model


# ====
# 3. MAIN
# ====

def main(seed: int | None = None) -> None:
    seed = Config.SEED if seed is None else seed

    print("\n" + "=" * 60)
    print(" MODEL TRAINING AND PERSISTENCE")
    print("=" * 60)
    print(f"Seed: {seed}")

    set_seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    reference = load_reference(seed)
    best_name = reference["selected_model"]
    print(f"Model selected by main.py (validation {RANKING_METRIC}): {best_name}")

    # ---- Same data, same dedup, same labels, same split as main.py ----
    texts = load_data(Config.DATA_PATH, sample_size=Config.SAMPLE_SIZE, seed=seed)
    texts, dedup_stats = maybe_deduplicate(texts)

    labels, _ = label_texts(texts)

    train_texts, val_texts, test_texts, y_train, y_val, y_test = split_data(
        texts, labels, seed
    )
    del texts, labels
    gc.collect()

    splits = {"train": len(y_train), "validation": len(y_val), "test": len(y_test)}
    check_splits(reference, splits)
    print(
        f"Train: {splits['train']:,}  Validation: {splits['validation']:,}  "
        f"Test: {splits['test']:,}  (matches metrics.json)"
    )

    # ---- Features: fitted on train only ----
    (
        X_train_tfidf, X_val_tfidf, X_test_tfidf,
        X_train_pca, X_val_pca, X_test_pca,
        tfidf_vectorizer, pca_model,
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

    n_tfidf = int(X_train_tfidf.shape[1])
    n_pca = int(X_train_pca.shape[1])

    X_train = combine_features(X_train_tfidf, X_train_pca)
    X_val = combine_features(X_val_tfidf, X_val_pca)
    X_test = combine_features(X_test_tfidf, X_test_pca)

    del X_train_tfidf, X_val_tfidf, X_test_tfidf, X_train_pca, X_val_pca, X_test_pca
    del train_texts, val_texts, test_texts
    gc.collect()

    # ---- Scaling (train only) and oversampling (train only) ----
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
    backend = "imbalanced-learn" if HAS_IMBLEARN else "numpy fallback"
    print(f"Oversampling: {len(y_train):,} -> {len(y_train_res):,} ({backend})")

    del X_train_scaled
    gc.collect()

    # ---- Train the selected model ----
    print(f"\nTraining {best_name}...")
    model = train_selected(best_name, seed, X_train_res, y_train_res, X_val_scaled, y_val)

    del X_train_res, y_train_res
    gc.collect()

    # ---- Threshold: selected on validation by train.select_threshold, frozen for test ----
    proba_val = _predict_proba(model, X_val_scaled)
    proba_test = _predict_proba(model, X_test_scaled)

    if proba_val is not None:
        threshold, val_score, _ = select_threshold(y_val, proba_val, metric=RANKING_METRIC)
        print(
            f"Decision threshold (validation {RANKING_METRIC} = {val_score:.4f}): "
            f"{threshold:.4f}"
        )
        val_pred = predict_with_threshold(proba_val, threshold)
        test_pred = predict_with_threshold(proba_test, threshold)
    else:
        # Same fallback as train.train_and_evaluate for models without probabilities
        threshold = 0.5
        print("Model has no predict_proba; using default threshold 0.5 (as in train.py).")
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)

    val_metrics = evaluate_model(y_val, val_pred, proba_val, threshold=threshold)
    test_metrics = evaluate_model(y_test, test_pred, proba_test, threshold=threshold)

    print(
        f"Test ({best_name}, frozen threshold): "
        f"macro-F1={test_metrics['macro_f1']:.4f}  "
        f"BalAcc={test_metrics['balanced_accuracy']:.4f}  "
        f"Spam F1={test_metrics['f1']:.4f}  "
        f"Genuine recall={test_metrics['genuine_recall']:.4f}"
    )

    consistent_val = check_against_reference(best_name, "validation", val_metrics, reference)
    consistent_test = check_against_reference(best_name, "test", test_metrics, reference)
    consistent = consistent_val and consistent_test

    if not consistent:
        print(
            "\n  [warn] The persisted model does not reproduce metrics.json. "
            "Do not use these artifacts for the paper until this is resolved."
        )

    # ---- Persist artifacts ----
    print("\nSaving artifacts...")
    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(pca_model, os.path.join(MODELS_DIR, "pca_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(model, os.path.join(MODELS_DIR, "best_model.joblib"))

    drop = {"fpr", "tpr", "feature_importances"}
    model_info = {
        "selected_model": best_name,
        "selection_metric": RANKING_METRIC,
        "selection_split": "validation",
        "decision_threshold": float(threshold),
        "threshold_selected_on": "validation",
        "seed": seed,
        "oversampling_backend": backend,
        "feature_layout": {
            "order": "TF-IDF then BERT-PCA",
            "tfidf": n_tfidf,
            "bert_pca": n_pca,
            "total": n_tfidf + n_pca,
        },
        "splits": splits,
        "dedup": dedup_stats,
        "config": config_snapshot(),
        "metrics": {
            "validation": {k: v for k, v in val_metrics.items() if k not in drop},
            "test": {k: v for k, v in test_metrics.items() if k not in drop},
        },
        "consistent_with_metrics_json": bool(consistent),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(os.path.join(MODELS_DIR, "model_info.json"), "w", encoding="utf-8") as file:
        json.dump(_json_safe(model_info), file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f" Saved directory: {MODELS_DIR}/")
    print("  - tfidf_vectorizer.joblib")
    print("  - pca_model.joblib")
    print("  - scaler.joblib")
    print("  - best_model.joblib")
    print("  - model_info.json  (selected model, frozen threshold, feature layout)")
    print("=" * 60 + "\n")


# ====
# 4. ENTRY POINT
# ====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persist the model selected by main.py")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Must match the seed used for the main.py run that wrote metrics.json.",
    )
    args = parser.parse_args()

    main(seed=args.seed)
