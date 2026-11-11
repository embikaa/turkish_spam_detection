"""train.py — model zoo, imbalance-aware evaluation, ANN early stopping, statistics.

Design rules enforced here:
  * one ranking metric everywhere (RANKING_METRIC), so the validation table,
    the test table and the plots cannot disagree about "the best model";
  * the ANN is early-stopped on validation macro-F1, never on accuracy, and the
    validation split is never oversampled;
  * the decision threshold is selected on the validation split and then frozen
    for the test split, so no test information leaks into model selection;
  * every metric that the paper reports is produced by evaluate_model, in one
    place, with identical definitions for all four models.
"""
from __future__ import annotations

# YENİ:
from copy import deepcopy
import gc
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# Single ranking metric. Every sort in this module and in visualize.py must use
# this key; changing it here changes the paper's "best model" everywhere.
RANKING_METRIC = "macro_f1"

# Metrics aggregated across seeds by summarize_seeds().
SEED_METRIC_KEYS = (
    "accuracy",
    "f1",
    "precision",
    "recall",
    "macro_f1",
    "balanced_accuracy",
    "genuine_f1",
    "auc",
)


# ====
# 1. MODEL ZOO
# ====

def get_models(seed: int = 42) -> Dict[str, Any]:
    """Four classifiers, identical hyper-parameters across seeds.

    The ANN keeps early_stopping=False on purpose: sklearn's internal early
    stopping would carve its own validation slice out of the already resampled
    training matrix, which is exactly the leakage pattern we removed.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    try:
        from lightgbm import LGBMClassifier

        lightgbm = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        lightgbm = None

    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            solver="saga",
            max_iter=500,
            random_state=seed,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=seed,
            n_jobs=-1,
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=300,
            early_stopping=False,
            batch_size=512,
            random_state=seed,
        ),
    }

    if lightgbm is not None:
        models["LightGBM"] = lightgbm

    return models


# ====
# 2. EVALUATION
# ====

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """All reported metrics for one model on one split.

    Positive class is spam (1). Keys "f1", "precision", "recall" keep their
    historical meaning (spam class) so that existing callers and the saved
    figures stay comparable; "macro_f1" and "balanced_accuracy" are the
    imbalance-aware additions used for ranking and for the paper's tables.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),

        "macro_f1": round(float(np.mean(f1_c)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),

        "genuine_precision": round(float(prec_c[0]), 4),
        "genuine_recall": round(float(rec_c[0]), 4),
        "genuine_f1": round(float(f1_c[0]), 4),
        "genuine_support": int(support_c[0]),

        "spam_support": int(support_c[1]),
        "n_samples": int(len(y_true)),
        "threshold": None if threshold is None else round(float(threshold), 4),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float).ravel()

        if np.isfinite(y_proba).all() and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            metrics["auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
            metrics["fpr"] = fpr
            metrics["tpr"] = tpr

    return metrics


def _predict_proba(model, X: np.ndarray) -> Optional[np.ndarray]:
    """Spam probability, or None when the estimator cannot produce one."""
    if not hasattr(model, "predict_proba"):
        return None

    try:
        proba = model.predict_proba(X)
    except (AttributeError, ValueError):
        return None

    if proba.ndim != 2 or proba.shape[1] < 2:
        return None

    return np.asarray(proba[:, 1], dtype=float)


# ====
# 3. DECISION THRESHOLD
# ====

def predict_with_threshold(
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Turn probabilities into labels at a fixed threshold."""
    return (np.asarray(y_proba, dtype=float).ravel() >= threshold).astype(int)


def select_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = RANKING_METRIC,
    n_grid: int = 201,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """Pick the decision threshold on the validation split.

    Sweeping every distinct probability would be O(n^2); a quantile grid of
    n_grid candidates covers the same range in O(n log n) and is stable for
    imbalanced data because the low quantiles concentrate where the spam mass
    is. The selected threshold is frozen and reused on the test split.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()

    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan"), []

    quantiles = np.linspace(0.50, 0.999, n_grid)
    candidates = np.unique(np.clip(np.quantile(y_proba, quantiles), 0.0, 1.0))
    candidates = np.unique(np.concatenate([candidates, [0.5]]))

    def score_at(threshold: float) -> float:
        y_pred = predict_with_threshold(y_proba, threshold)

        if metric == "macro_f1":
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if metric == "spam_f1":
            return float(f1_score(y_true, y_pred, zero_division=0))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        raise ValueError(f"Unsupported threshold metric: {metric}")

    table = [(float(t), score_at(float(t))) for t in candidates]
    best_threshold, best_score = max(table, key=lambda item: item[1])

    return float(best_threshold), float(best_score), table


# ====
# 4. ANN EARLY STOPPING
# ====

def fit_ann_with_validation(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 300,
    patience: int = 10,
    tol: float = 1e-4,
    metric: str = RANKING_METRIC,
    min_epochs: int = 5,
    verbose: bool = True,
) -> Tuple[Any, Dict[str, List[float]]]:
    """One partial_fit per epoch; best weights by validation macro-F1.

    Accuracy is deliberately not used: with a 1.3% spam rate the accuracy
    optimum is the all-genuine predictor, so an accuracy-stopped ANN can return
    weights from an epoch in which spam was never learned. macro-F1 weights both
    classes equally and is the same metric used for model ranking.

    Returns the best model and the per-epoch history, so the training curve can
    be plotted and the stopping point can be audited.
    """
    classes = np.unique(y_train)

    if len(classes) < 2:
        raise ValueError("The training split contains a single class.")

    best_model = deepcopy(model)
    best_score = -np.inf
    no_improvement = 0
    stopped_epoch = max_epochs

    history: Dict[str, List[float]] = {
        "epoch": [],
        "val_macro_f1": [],
        "val_balanced_accuracy": [],
        "val_spam_f1": [],
    }

    for epoch in range(max_epochs):
        model.partial_fit(X_train, y_train, classes=classes)

        y_val_pred = model.predict(X_val)

        macro_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
        balanced_acc = float(balanced_accuracy_score(y_val, y_val_pred))
        spam_f1 = float(f1_score(y_val, y_val_pred, zero_division=0))

        history["epoch"].append(epoch + 1)
        history["val_macro_f1"].append(round(macro_f1, 6))
        history["val_balanced_accuracy"].append(round(balanced_acc, 6))
        history["val_spam_f1"].append(round(spam_f1, 6))

        if metric == "macro_f1":
            score = macro_f1
        elif metric == "balanced_accuracy":
            score = balanced_acc
        elif metric == "spam_f1":
            score = spam_f1
        else:
            raise ValueError(f"Unsupported early-stopping metric: {metric}")

        if score > best_score:
            best_score = score
            best_model = deepcopy(model)

            # tol only governs the patience counter, not which weights are kept.
            if score >= best_score_previous(best_score, tol):
                no_improvement = 0
            else:
                no_improvement += 1
        else:
            no_improvement += 1

        if epoch + 1 >= min_epochs and no_improvement >= patience:
            stopped_epoch = epoch + 1
            break

    if verbose:
        print(
            f"  ANN early stopping: epoch={stopped_epoch}, "
            f"best validation {metric}={best_score:.4f}"
        )

    history["stopped_epoch"] = [stopped_epoch]
    history["best_score"] = [round(float(best_score), 6)]

    return best_model, history


def best_score_previous(current_best: float, tol: float) -> float:
    """Helper kept separate so the patience rule stays readable."""
    return current_best - tol


# ====
# 5. TRAIN AND EVALUATE
# ====

# YENİ:
def train_and_evaluate(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    threshold_metric: str = RANKING_METRIC,
    verbose: bool = True,
    return_predictions: bool = False,
) -> Union[
    Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]],
    Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, np.ndarray]],
]:
    """Train every model, select its threshold on validation, score test once.

    Model selection uses validation_results only. The test split is touched
    exactly once per model, after the threshold is frozen, so no test
    information can influence any choice.

    return_predictions=True additionally returns each successful model's test
    predictions (name -> int array). Bulgu 4's paired McNemar/bootstrap tests
    in run_seeds.py need them; main.py's call is unchanged (default False).
    """
    models = get_models(seed)
    validation_results: Dict[str, Dict[str, Any]] = {}
    test_results: Dict[str, Dict[str, Any]] = {}
    test_predictions: Dict[str, np.ndarray] = {}

    print(f"\nTraining {len(models)} models...\n")

    for name, model in tqdm(models.items(), desc="Training", unit="model"):
        start_time = time.time()

        try:
            ann_history: Optional[Dict[str, List[float]]] = None

            if name == "ANN":
                model, ann_history = fit_ann_with_validation(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    metric=threshold_metric,
                    verbose=verbose,
                )
            else:
                model.fit(X_train, y_train)

            elapsed = round(time.time() - start_time, 2)

            val_proba = _predict_proba(model, X_val)
            test_proba = _predict_proba(model, X_test)

            if val_proba is not None:
                threshold, threshold_score, _ = select_threshold(
                    y_val, val_proba, metric=threshold_metric
                )
                val_pred = predict_with_threshold(val_proba, threshold)
                test_pred = predict_with_threshold(test_proba, threshold)
            else:
                threshold, threshold_score = 0.5, float("nan")
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)

            val_metrics = evaluate_model(y_val, val_pred, val_proba, threshold)
            test_metrics = evaluate_model(y_test, test_pred, test_proba, threshold)

            for metrics in (val_metrics, test_metrics):
                metrics["time"] = elapsed
                metrics["threshold_metric"] = threshold_metric
                metrics["threshold_score"] = (
                    None if np.isnan(threshold_score) else round(threshold_score, 4)
                )
                metrics["n_features"] = int(X_train.shape[1])

                if hasattr(model, "feature_importances_"):
                    metrics["feature_importances"] = np.asarray(
                        model.feature_importances_, dtype=float
                    )

            if ann_history is not None:
                val_metrics["ann_history"] = ann_history
                test_metrics["ann_history"] = ann_history

            validation_results[name] = val_metrics
            test_results[name] = test_metrics

            if return_predictions:
                test_predictions[name] = np.asarray(
                    test_pred, dtype=np.int64
                ).ravel()

        except Exception as exc:  # noqa: BLE001 - one model must not kill the run
            message = f"{type(exc).__name__}: {exc}"
            validation_results[name] = {"error": message}
            test_results[name] = {"error": message}

            if verbose:
                print(f"  [warn] {name} failed: {message}")

        finally:
            del model
            gc.collect()

    if return_predictions:
        return validation_results, test_results, test_predictions

    return validation_results, test_results


# ====
# 6. REPORTING
# ====

def _ranked(
    results: Dict[str, Dict[str, Any]],
    metric: str = RANKING_METRIC,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Successful models, best first, by a single metric."""
    successful = [
        (name, metrics)
        for name, metrics in results.items()
        if "error" not in metrics
    ]
    successful.sort(key=lambda item: item[1].get(metric, 0.0), reverse=True)
    return successful


def print_results(
    results: Dict[str, Dict[str, Any]],
    metric: str = RANKING_METRIC,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Validation table, ranked by RANKING_METRIC."""
    ranked = _ranked(results, metric)
    failed = [(n, m) for n, m in results.items() if "error" in m]

    print("\n" + "=" * 100)
    print(f" VALIDATION RESULTS — ranked by {metric} ")
    print("=" * 100)
    print(
        f"{'Rank':<5} {'Model':<22} {'MacroF1':>9} {'SpamF1':>9} {'BalAcc':>9} "
        f"{'Prec':>8} {'Rec':>8} {'Acc':>8} {'Thr':>7} {'Time':>8}"
    )
    print("-" * 100)

    for rank, (name, m) in enumerate(ranked, 1):
        threshold = m.get("threshold")
        threshold_text = f"{threshold:.3f}" if isinstance(threshold, float) else "N/A"

        print(
            f"{rank:<5} {name:<22} {m['macro_f1']:>9.4f} {m['f1']:>9.4f} "
            f"{m['balanced_accuracy']:>9.4f} {m['precision']:>8.4f} "
            f"{m['recall']:>8.4f} {m['accuracy']:>8.4f} "
            f"{threshold_text:>7} {m['time']:>7.1f}s"
        )

    for name, m in failed:
        print(f"{'--':<5} {name:<22} ERROR: {m['error']}")

    print("=" * 100)

    if ranked:
        best_name, best_metrics = ranked[0]
        print(
            f"\nBest model on validation: {best_name} "
            f"({metric} = {best_metrics[metric]:.4f}, "
            f"threshold = {best_metrics.get('threshold')})"
        )

    return ranked


def print_test_results(
    test_results: Dict[str, Dict[str, Any]],
    validation_results: Dict[str, Dict[str, Any]],
    metric: str = RANKING_METRIC,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Test table in validation order, so the two tables cannot disagree."""
    order = _ranked(validation_results, metric)

    print(f"\nTEST RESULTS — ordered by validation {metric}")
    print(
        f"{'Model':<22} {'Acc':>8} {'SpamF1':>8} {'MacroF1':>8} {'BalAcc':>8} "
        f"{'Gen.P':>8} {'Gen.R':>8} {'Gen.F1':>8} {'AUC':>8} {'Thr':>7}"
    )
    print("-" * 108)

    printed: List[Tuple[str, Dict[str, Any]]] = []

    for name, _ in order:
        metrics = test_results.get(name, {"error": "missing"})

        if "error" in metrics:
            print(f"{name:<22} ERROR: {metrics['error']}")
            continue

        auc = metrics.get("auc")
        auc_text = f"{auc:.4f}" if auc is not None else "N/A"
        threshold = metrics.get("threshold")
        threshold_text = f"{threshold:.3f}" if isinstance(threshold, float) else "N/A"

        print(
            f"{name:<22} {metrics['accuracy']:>8.4f} {metrics['f1']:>8.4f} "
            f"{metrics['macro_f1']:>8.4f} {metrics['balanced_accuracy']:>8.4f} "
            f"{metrics['genuine_precision']:>8.4f} "
            f"{metrics['genuine_recall']:>8.4f} "
            f"{metrics['genuine_f1']:>8.4f} {auc_text:>8} {threshold_text:>7}"
        )
        print(
            f"  CM [[TN, FP], [FN, TP]] = {metrics['cm']}  "
            f"(genuine support {metrics['genuine_support']}, "
            f"spam support {metrics['spam_support']})"
        )

        printed.append((name, metrics))

    failed = [(n, m) for n, m in test_results.items() if "error" in m]
    for name, m in failed:
        print(f"{name:<22} ERROR: {m['error']}")

    return printed


# ====
# 7. STATISTICS (Bulgu 4 / Bulgu 8)
# ====

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Stratified bootstrap confidence interval for one metric.

    Resampling is done within each class so that every replicate keeps the
    original spam rate; an unstratified bootstrap on 1.3% prevalence produces
    replicates with zero spam and an undefined macro-F1.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    if metric_fn is None:
        metric_fn = lambda t, p: float(  # noqa: E731
            f1_score(t, p, average="macro", zero_division=0)
        )

    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y_true == 1)
    idx_neg = np.flatnonzero(y_true == 0)

    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}

    scores = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        take_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        take_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        idx = np.concatenate([take_pos, take_neg])
        scores[i] = metric_fn(y_true[idx], y_pred[idx])

    low, high = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std(ddof=1)), 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
        "n_boot": int(n_boot),
    }


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    continuity: bool = True,
) -> Dict[str, Any]:
    """McNemar test on paired correctness, for the paper's model comparison."""
    y_true = np.asarray(y_true).astype(int).ravel()
    pred_a = np.asarray(pred_a).astype(int).ravel()
    pred_b = np.asarray(pred_b).astype(int).ravel()

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    n_a_only = int(np.sum(a_correct & ~b_correct))
    n_b_only = int(np.sum(~a_correct & b_correct))
    discordant = n_a_only + n_b_only

    if discordant == 0:
        return {
            "n_a_only": 0, "n_b_only": 0, "discordant": 0,
            "statistic": 0.0, "p_value": 1.0, "significant": False,
        }

    if continuity:
        statistic = (abs(n_a_only - n_b_only) - 1) ** 2 / discordant
    else:
        statistic = (n_a_only - n_b_only) ** 2 / discordant

    p_value = float(math.erfc(math.sqrt(statistic / 2.0)))

    return {
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "discordant": discordant,
        "statistic": round(float(statistic), 4),
        "p_value": round(p_value, 6),
        "significant": bool(p_value < 0.05),
    }


def summarize_seeds(
    seed_results: Sequence[Dict[str, Dict[str, Any]]],
    metric_keys: Sequence[str] = SEED_METRIC_KEYS,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mean and standard deviation across seeds, per model and metric.

    Used by run_seeds.py to turn five single-seed runs into the
    "mean ± std" rows the referee asks for in §4.
    """
    if not seed_results:
        return {}

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for name in seed_results[0]:
        per_metric: Dict[str, Dict[str, float]] = {}

        for key in metric_keys:
            values = [
                run[name][key]
                for run in seed_results
                if key in run.get(name, {})
                and run[name].get(key) is not None
                and not isinstance(run[name].get(key), str)
            ]

            if not values:
                continue

            array = np.asarray(values, dtype=float)
            std = float(array.std(ddof=1)) if len(array) > 1 else 0.0

            per_metric[key] = {
                "mean": round(float(array.mean()), 4),
                "std": round(std, 4),
                "n": int(len(array)),
            }

        if per_metric:
            summary[name] = per_metric

    return summary


__all__ = [
    "RANKING_METRIC",
    "SEED_METRIC_KEYS",
    "get_models",
    "evaluate_model",
    "predict_with_threshold",
    "select_threshold",
    "fit_ann_with_validation",
    "train_and_evaluate",
    "print_results",
    "print_test_results",
    "bootstrap_ci",
    "mcnemar_test",
    "summarize_seeds",
]
