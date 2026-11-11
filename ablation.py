"""ablation.py — feature-set ablation (TF-IDF only / BERT only / hybrid) on the ANN.

Referee findings handled in this module:
  * Bulgu 3 / 5: the split, the deduplication, the oversampling and the seed
    handling are imported from main.py, so the ablation runs on exactly the
    same train/validation/test split as every other number in the paper;
  * Bulgu 4: the seed is a --seed argument (default Config.SEED) and every
    number printed here is also written to results/ablation.json;
  * Bulgu 7: variants are ranked by train.RANKING_METRIC (macro_f1), the
    single ranking key used across the pipeline;
  * Section 6: the decision threshold is selected on the validation split and
    frozen before the test split is scored;
  * Section 3.4: macro-F1, balanced accuracy and genuine-class metrics are
    reported next to the spam-class metrics.

Usage:
    python main.py --seed 42        # run first; defines the canonical split
    python ablation.py --seed 42
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import warnings
from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler

from config import Config
from features import combine_features, create_three_way_features
from labeling import (
    get_labeling_diagnostics,
    label_texts,
    print_labeling_diagnostics,
)
from main import (
    HAS_IMBLEARN,
    _json_safe,
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

ABLATION_PATH = os.path.join(Config.RESULTS_DIR, "ablation.json")


# ====
# 1. ONE VARIANT = ONE FIT
# ====

def train_eval(X_tr, X_val, X_te, y_tr, y_val, y_te, label, seed):
    """Fit the ANN on one feature set with the exact main.py protocol.

    Scaling is fitted on the training split only, oversampling touches the
    training split only, and the threshold is picked on validation and frozen
    for test — the same order main.py and save_model.py use.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    X_res, y_res = resample_training_set(
        X_tr_s,
        y_tr,
        ratio=Config.OVERSAMPLING_RATIO,
        seed=seed,
    )
    X_res = np.asarray(X_res, dtype=np.float32)

    print(f"\n{label}")
    print(
        f"  Oversampling: {len(y_tr):,} -> {len(y_res):,} "
        f"({'imbalanced-learn' if HAS_IMBLEARN else 'numpy fallback'})"
    )

    ann = get_models(seed)["ANN"]
    ann = fit_ann_with_validation(ann, X_res, y_res, X_val_s, y_val)

    val_proba = _predict_proba(ann, X_val_s)
    test_proba = _predict_proba(ann, X_te_s)

    threshold, threshold_score, _ = select_threshold(
        y_val, val_proba, metric=RANKING_METRIC
    )
    val_pred = predict_with_threshold(val_proba, threshold)
    test_pred = predict_with_threshold(test_proba, threshold)

    val_metrics = evaluate_model(y_val, val_pred, val_proba, threshold)
    test_metrics = evaluate_model(y_test, test_pred, test_proba, threshold)

    print(f"  Frozen threshold ({RANKING_METRIC} on validation): {threshold:.2f}")
    for split_name, metrics in (("Validation", val_metrics), ("Test", test_metrics)):
        print(
            f"  {split_name}: "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.4f}, "
            f"spam_f1={metrics['f1']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"genuine_recall={metrics['genuine_recall']:.4f}, "
            f"AUC={metrics['auc']:.4f}"
        )
        print(f"  {split_name} confusion matrix: {metrics['cm']}")

    return {
        "variant": label,
        "threshold": threshold,
        "threshold_validation_score": threshold_score,
        "validation": val_metrics,
        "test": test_metrics,
    }


# ====
# 2. ENTRY POINT
# ====

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature-set ablation (TF-IDF only / BERT only / hybrid) on the ANN."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override Config.SEED (use the same seed as the main.py run you compare with).",
    )
    args = parser.parse_args()
    seed = Config.SEED if args.seed is None else args.seed

    set_seed(seed)

    # --- data: identical path to main.py ---
    texts = load_data(Config.DATA_PATH, seed=seed)
    texts, dedup_stats = maybe_deduplicate(texts)

    labels, triggered_rules = label_texts(texts)
    diagnostics = get_labeling_diagnostics(labels, triggered_rules)
    print_labeling_diagnostics(diagnostics)

    train_texts, val_texts, test_texts, y_train, y_val, y_test = split_data(
        texts, labels, seed
    )

    del texts, labels, triggered_rules
    gc.collect()

    print(
        f"Train: {len(train_texts):,}  "
        f"Validation: {len(val_texts):,}  "
        f"Test: {len(test_texts):,}"
    )

    # --- features: fitted on the training split only, as in main.py ---
    (
        X_tr_tfidf,
        X_val_tfidf,
        X_te_tfidf,
        X_tr_bert_pca,
        X_val_bert_pca,
        X_te_bert_pca,
        _,
        _,
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

    print("\n" + "=" * 50)
    print(" ABLATION STUDY")
    print("=" * 50)

    results = [
        train_eval(
            X_tr_tfidf, X_val_tfidf, X_te_tfidf,
            y_train, y_val, y_test,
            "TF-IDF Only", seed,
        ),
        train_eval(
            X_tr_bert_pca, X_val_bert_pca, X_te_bert_pca,
            y_train, y_val, y_test,
            "BERT Only", seed,
        ),
    ]

    X_tr_h = combine_features(X_tr_tfidf, X_tr_bert_pca)
    X_val_h = combine_features(X_val_tfidf, X_val_bert_pca)
    X_te_h = combine_features(X_te_tfidf, X_te_bert_pca)

    results.append(
        train_eval(
            X_tr_h, X_val_h, X_te_h,
            y_train, y_val, y_test,
            "TF-IDF + BERT (Hybrid)", seed,
        )
    )

    ranked = sorted(
        results,
        key=lambda r: r["validation"][RANKING_METRIC],
        reverse=True,
    )
    print(f"\nRanking by validation {RANKING_METRIC}:")
    for rank, r in enumerate(ranked, 1):
        print(
            f"  {rank}. {r['variant']}: "
            f"val={r['validation'][RANKING_METRIC]:.4f}  "
            f"test={r['test'][RANKING_METRIC]:.4f}"
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "ranking_metric": RANKING_METRIC,
        "oversampling_ratio": Config.OVERSAMPLING_RATIO,
        "labeling": {
            "total_reviews": diagnostics["total_reviews"],
            "class_distribution": diagnostics["class_distribution"],
            "class_percentages": diagnostics["class_percentages"],
        },
        "deduplication": dedup_stats,
        "best_variant_on_validation": ranked[0]["variant"],
        "variants": results,
    }

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    with open(ABLATION_PATH, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)

    print(f"\nAblation results written to {ABLATION_PATH}")


if __name__ == "__main__":
    main()
