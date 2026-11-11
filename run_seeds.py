"""run_seeds.py — Bulgu 4: multi-seed replication with mean ± std and
between-model significance tests (McNemar + paired bootstrap).

Referee findings handled in this module:
  * Bulgu 4: the full pipeline is repeated with at least 5 different seeds
    and the TEST metrics of every model are reported as mean ± std;
    between-model significance is assessed with train.mcnemar_test and a
    paired stratified bootstrap on the test split;
  * Single pipeline: every step (loading, deduplication, labeling, split,
    features, scaling, oversampling, training, threshold selection, test
    scoring) is the same function call main.py makes, in the same order.
    The training loop is NOT duplicated here — run_seeds.py calls
    train.train_and_evaluate directly, and the mean ± std table is produced
    by train.summarize_seeds;
  * Validation integrity: per seed, every model's decision threshold is
    selected on the validation split and frozen before the test split is
    scored exactly once — inside train_and_evaluate, as in main.py;
  * Bulgu 7: models are ranked per seed by train.RANKING_METRIC (macro_f1).

Only results/seeds.json is written; metrics.json and models/ are untouched.

Usage:
    python run_seeds.py                          # Config.SEEDS
    python run_seeds.py --seeds 42,0,1,2,3       # override the seed list
    python run_seeds.py --seeds 42 --n-bootstrap 50   # smoke test

Each seed repeats the BERT extraction and the full training; expect roughly
len(seeds) x (one `python main.py --seed` run) of total wall-clock time.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import warnings
from datetime import datetime
from itertools import combinations

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
    SEED_METRIC_KEYS,
    mcnemar_test,
    summarize_seeds,
    train_and_evaluate,
)

warnings.filterwarnings("ignore")

SEEDS_PATH = os.path.join(Config.RESULTS_DIR, "seeds.json")

# Bulgu 4's mean ± std table also reports the genuine-class recall, which is
# not part of train.SEED_METRIC_KEYS; everything else is inherited.
SUMMARY_METRIC_KEYS = SEED_METRIC_KEYS + ("genuine_recall",)

DEFAULT_SEEDS = [42, 0, 1, 2, 3]
DEFAULT_N_BOOTSTRAP = 1000

_SLIM_KEYS = ("fpr", "tpr", "feature_importances", "ann_history")


def _slim(metrics: dict) -> dict:
    """Drop the bulky arrays (ROC curve, feature importances, ANN history)
    before serialisation; every reported number stays. The canonical seed's
    full metrics, including those arrays, already live in metrics.json."""
    return {k: v for k, v in metrics.items() if k not in _SLIM_KEYS}


# ====
# 1. ONE SEED = ONE COMPLETE REPLICATION OF main.py
# ====

def run_one_seed(seed: int, verbose_labeling: bool = False):
    """Run the canonical pipeline for one seed.

    Returns (seed_record, y_test, test_predictions). test_predictions maps
    model name -> test predictions and is kept in memory only: the paired
    significance tests need them, seeds.json does not.
    """
    set_seed(seed)

    texts = load_data(Config.DATA_PATH, sample_size=Config.SAMPLE_SIZE, seed=seed)
    texts, dedup_stats = maybe_deduplicate(texts)

    labels, triggered_rules = label_texts(texts)
    diagnostics = get_labeling_diagnostics(labels, triggered_rules)
    if verbose_labeling:
        # The rules are deterministic, so the labeling diagnostics are
        # identical for every seed; print them once.
        print_labeling_diagnostics(diagnostics)

    train_texts, val_texts, test_texts, y_train, y_val, y_test = split_data(
        texts, labels, seed
    )
    del texts, labels, triggered_rules
    gc.collect()

    print(
        f"[seed {seed}] Train: {len(y_train):,}  "
        f"Validation: {len(y_val):,}  Test: {len(y_test):,}"
    )

    (
        X_train_tfidf, X_val_tfidf, X_test_tfidf,
        X_train_pca, X_val_pca, X_test_pca,
        _, _,
    ) = create_three_way_features(
        train_texts, val_texts, test_texts,
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
    del X_train_tfidf, X_val_tfidf, X_test_tfidf, X_train_pca, X_val_pca, X_test_pca
    gc.collect()

    # Fitted on the training split only; float32 keeps peak memory at the
    # canonical run's level. Same casts, same order as main.py.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    del X_train, X_val, X_test, scaler
    gc.collect()

    X_train_res, y_train_res = resample_training_set(
        X_train_scaled, y_train, ratio=Config.OVERSAMPLING_RATIO, seed=seed
    )
    X_train_res = np.asarray(X_train_res, dtype=np.float32)
    del X_train_scaled
    gc.collect()

    # The whole training/threshold/test protocol lives in train.py — the
    # exact function main.py calls, so the multi-seed run cannot drift.
    validation_results, test_results, test_predictions = train_and_evaluate(
        X_train_res, X_val_scaled, X_test_scaled,
        y_train_res, y_val, y_test,
        seed=seed, return_predictions=True,
    )

    del X_train_res, X_val_scaled, X_test_scaled, y_train_res
    gc.collect()

    seed_record = {
        "seed": seed,
        "deduplication": dedup_stats,
        "split_sizes": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "models": {
            name: {
                "threshold": m.get("threshold"),
                "validation": _slim(validation_results[name]),
                "test": _slim(test_results[name]),
            }
            for name, m in validation_results.items()
        },
    }

    successful = {
        name: m["validation"]
        for name, m in seed_record["models"].items()
        if "error" not in m["validation"]
    }
    seed_record["selected_model"] = (
        max(successful, key=lambda n: successful[n][RANKING_METRIC])
        if successful
        else None
    )

    for name in successful:
        test = seed_record["models"][name]["test"]
        print(
            f"[seed {seed}] {name}: test macro_f1={test['macro_f1']:.4f}  "
            f"balanced_acc={test['balanced_accuracy']:.4f}  "
            f"genuine_recall={test['genuine_recall']:.4f}"
        )

    return seed_record, y_test, test_predictions


# ====
# 2. SIGNIFICANCE TESTS (Bulgu 4: McNemar + paired bootstrap)
# ====

def _macro_f1(y, pred):
    """Exact macro-F1 for 0/1 labels from the 2x2 contingency table."""
    m = np.bincount(
        np.asarray(y, dtype=np.int64) * 2 + np.asarray(pred, dtype=np.int64),
        minlength=4,
    ).reshape(2, 2)
    f1s = []
    for c in (0, 1):
        denom = 2 * m[c, c] + m[1 - c, c] + m[c, 1 - c]
        f1s.append(0.0 if denom == 0 else 2.0 * m[c, c] / denom)
    return float(np.mean(f1s))


def paired_bootstrap_diff(y_true, pred_a, pred_b, n_boot, seed):
    """Paired stratified bootstrap for macro_f1(A) - macro_f1(B) on one split.

    Same rationale as train.bootstrap_ci: resampling within each class keeps
    the spam rate of every replicate, so macro-F1 stays defined. Both models
    are scored on the SAME replicate indices, so the difference is paired.
    Returns the 95% percentile CI and a two-sided bootstrap p-value.
    """
    y = np.asarray(y_true).astype(int).ravel()
    a = np.asarray(pred_a).astype(int).ravel()
    b = np.asarray(pred_b).astype(int).ravel()

    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)

    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return {
            "ci95_low": float("nan"), "ci95_high": float("nan"),
            "p_value": float("nan"), "n_boot": 0,
        }

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        take = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True),
        ])
        diffs[i] = _macro_f1(y[take], a[take]) - _macro_f1(y[take], b[take])

    low, high = np.percentile(diffs, [2.5, 97.5])
    p_value = 2.0 * min(np.mean(diffs > 0), np.mean(diffs < 0))

    return {
        "ci95_low": round(float(low), 4),
        "ci95_high": round(float(high), 4),
        "p_value": round(float(min(1.0, p_value)), 4),
        "n_boot": int(n_boot),
    }


def significance_tests(per_seed_records, preds_by_seed, y_by_seed, model_names, n_boot):
    mcnemar = {}
    bootstrap = {}

    for name_a, name_b in combinations(model_names, 2):
        pair = f"{name_a} vs {name_b}"
        rows_m = []
        rows_b = []

        for rec in per_seed_records:
            seed = rec["seed"]
            preds = preds_by_seed[seed]
            if name_a not in preds or name_b not in preds:
                continue
            y = y_by_seed[seed]

            result = mcnemar_test(y, preds[name_a], preds[name_b])
            rows_m.append({
                "seed": seed,
                "n_a_only": result["n_a_only"],
                "n_b_only": result["n_b_only"],
                "p_value": result["p_value"],
            })

            rows_b.append({
                "seed": seed,
                **paired_bootstrap_diff(
                    y, preds[name_a], preds[name_b], n_boot, seed
                ),
            })

        mcnemar[pair] = {
            "test": "McNemar (continuity-corrected chi2, df=1)",
            "per_seed": rows_m,
            "n_seeds_p_below_0.05": sum(
                1 for r in rows_m if r["p_value"] < 0.05
            ),
        }
        bootstrap[pair] = {
            "test": "paired stratified bootstrap, macro_f1 difference",
            "per_seed": rows_b,
            "n_seeds_ci_excludes_0": sum(
                1 for r in rows_b
                if r["ci95_low"] > 0 or r["ci95_high"] < 0
            ),
        }

    return mcnemar, bootstrap


# ====
# 3. ENTRY POINT
# ====

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulgu 4: multi-seed replication with mean ± std and "
        "McNemar/bootstrap significance tests."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list, e.g. 42,0,1,2,3 (default: Config.SEEDS).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=None,
        help="Bootstrap resamples for the paired macro-F1 difference test.",
    )
    args = parser.parse_args()

    seeds = (
        [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else list(getattr(Config, "SEEDS", DEFAULT_SEEDS))
    )
    n_boot = (
        args.n_bootstrap
        if args.n_bootstrap is not None
        else getattr(Config, "N_BOOTSTRAP", DEFAULT_N_BOOTSTRAP)
    )

    if len(seeds) < 5:
        print(
            f"Warning: Bulgu 4 asks for at least 5 seeds; got {len(seeds)}. "
            "Use this only as a smoke test."
        )

    per_seed_records = []
    preds_by_seed = {}
    y_by_seed = {}

    for i, seed in enumerate(seeds):
        print("\n" + "=" * 60)
        print(f" SEED {seed}  ({i + 1}/{len(seeds)})")
        print("=" * 60)
        record, y_test, preds = run_one_seed(seed, verbose_labeling=(i == 0))
        per_seed_records.append(record)
        preds_by_seed[seed] = preds
        y_by_seed[seed] = np.asarray(y_test).astype(int).ravel()

    # train.summarize_seeds: mean/std across seeds, per model and metric.
    test_dicts = [
        {name: m["test"] for name, m in rec["models"].items()}
        for rec in per_seed_records
    ]
    summary = summarize_seeds(test_dicts, metric_keys=SUMMARY_METRIC_KEYS)

    selected = [
        rec["selected_model"] for rec in per_seed_records if rec["selected_model"]
    ]
    selected_counts = {name: selected.count(name) for name in sorted(set(selected))}

    print("\n" + "=" * 78)
    print(f" MULTI-SEED SUMMARY — TEST SPLIT, mean ± std over {len(seeds)} seeds")
    print("=" * 78)
    for name, m in summary.items():
        def cell(key):
            return f"{m[key]['mean']:.4f}±{m[key]['std']:.4f}" if key in m else "N/A"

        print(
            f"{name:<22} macro_f1={cell('macro_f1')}  "
            f"bal_acc={cell('balanced_accuracy')}  "
            f"spam_f1={cell('f1')}  gen_recall={cell('genuine_recall')}"
        )
    print(
        f"\nSelected model per seed (by validation {RANKING_METRIC}): "
        f"{selected_counts}"
    )

    print("\nRunning significance tests (McNemar + paired bootstrap)...")
    mcnemar, bootstrap = significance_tests(
        per_seed_records, preds_by_seed, y_by_seed, list(summary), n_boot
    )

    for pair, res in mcnemar.items():
        print(
            f"  McNemar {pair}: seeds p<0.05 = "
            f"{res['n_seeds_p_below_0.05']}/{len(res['per_seed'])}"
        )
    for pair, res in bootstrap.items():
        print(
            f"  Bootstrap {pair}: CI excludes 0 in "
            f"{res['n_seeds_ci_excludes_0']}/{len(res['per_seed'])} seeds"
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ranking_metric": RANKING_METRIC,
        "config": config_snapshot(),
        "seeds": seeds,
        "n_bootstrap": n_boot,
        "oversampling_ratio": Config.OVERSAMPLING_RATIO,
        "per_seed": per_seed_records,
        "summary": {
            "test_mean_std": summary,
            "selected_model_counts": selected_counts,
            "mcnemar": mcnemar,
            "bootstrap": bootstrap,
        },
    }

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    with open(SEEDS_PATH, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)

    print(f"\nMulti-seed results written to {SEEDS_PATH}")


if __name__ == "__main__":
    main()
