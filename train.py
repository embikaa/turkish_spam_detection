"""Model training and evaluation pipeline."""
import gc
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


def get_models(seed: int = 42) -> Dict[str, Any]:
    """
    Initialize five classification models for evaluation.

    Models:
    - Logistic Regression: Fast baseline
    - ANN (MLP): Three-layer neural network with adaptive learning
    - CART: Decision tree with depth limit
    - Random Forest: Ensemble for feature importance analysis
    - LightGBM: Gradient boosting with fast convergence

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping model names to initialized classifiers.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from lightgbm import LGBMClassifier

    models = {
        "Logistic Regression": LogisticRegression(
            solver="saga", max_iter=500, random_state=seed, n_jobs=-1
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=300,
            early_stopping=True,
            batch_size=512,
            random_state=seed,
        ),
        "CART": DecisionTreeClassifier(
            max_depth=20, random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=25, random_state=seed, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150, learning_rate=0.1, random_state=seed, n_jobs=-1, verbose=-1
        ),
    }

    return models


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute classification metrics for model evaluation.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional, for AUC calculation).

    Returns:
        Dictionary containing accuracy, F1, precision, recall,
        confusion matrix, and AUC (if probabilities provided).
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        metrics["auc"] = round(roc_auc_score(y_true, y_proba), 4)

    return metrics


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all classification models.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training labels.
        y_test: Test labels.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping model names to their evaluation metrics.
    """
    models = get_models(seed)
    results = {}

    print(f"\nTraining {len(models)} models...\n")

    for name, model in tqdm(models.items(), desc="Training", unit="model"):
        start_time = time.time()

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]

            metrics = evaluate_model(y_test, y_pred, y_proba)
            metrics["time"] = round(time.time() - start_time, 2)
            results[name] = metrics

        except Exception as e:
            results[name] = {"error": str(e)}

        del model
        gc.collect()

    return results


def print_results(results: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Print model evaluation results as a formatted table.

    Args:
        results: Dictionary of model results from train_and_evaluate().

    Returns:
        List of (model_name, metrics) tuples sorted by F1 score (descending).
    """
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("f1", 0),
        reverse=True
    )

    print("\n" + "=" * 85)
    print(" MODEL EVALUATION RESULTS ")
    print("=" * 85)
    print(f"{'Rank':<5} {'Model':<22} {'Accuracy':>10} {'F1':>10} {'Precision':>12} {'Recall':>10} {'Time':>8}")
    print("-" * 85)

    for i, (name, m) in enumerate(sorted_results, 1):
        if "error" in m:
            print(f"{i:<5} {name:<22} {'ERROR':>10}")
            continue

        print(
            f"{i:<5} {name:<22} {m['accuracy']:>10.4f} {m['f1']:>10.4f} "
            f"{m['precision']:>12.4f} {m['recall']:>10.4f} {m['time']:>7.1f}s"
        )

    print("=" * 85)

    if sorted_results and "error" not in sorted_results[0][1]:
        best_name = sorted_results[0][0]
        best_f1 = sorted_results[0][1].get("f1", 0)
        print(f"\nBest model: {best_name} (F1 Score: {best_f1:.4f})")

    return sorted_results
