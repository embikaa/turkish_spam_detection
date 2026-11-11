"""explainability.py — SHAP attribution for the model selected by main.py.

Referee findings handled in this module:
  * Bulgu 9 / Section 7.2: the method is stated exactly. Shapley values are
    estimated with shap.KernelExplainer (model-agnostic). This is NOT
    permutation importance; the figure axis and the summary file say so;
  * Section 7.2: the analysis covers the full hybrid space (TF-IDF + BERT-PCA),
    not only the TF-IDF subspace, and the total contribution of each block is
    reported;
  * Section 7.2: ranking stability is quantified with bootstrap 95% CIs over
    the explained reviews and the top-N inclusion frequency of each feature;
  * Section 7.2: the top-N list is printed numbered and stored in
    results/shap_summary.json, so the paper can quote the exact list;
  * Bulgu 4: model, frozen threshold and feature layout are read from models/
    and checked against results/metrics.json; nothing is retrained or re-fitted;
  * Bulgu 5 / Section 3.4: same data, dedup, labels and split as main.py
    (main.split_data); the persisted TF-IDF, PCA and scaler are only applied;
    the SHAP background is drawn from the training split;
  * Section 9.3: the figure is saved at 600 dpi.

Random Forest MDI (impurity) importance is a different method. It is produced
by visualize.py and must not be described as SHAP.

Usage:
    python main.py --seed 42
    python save_model.py --seed 42
    python explainability.py --seed 42
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from config import Config
from features import combine_features, create_bert_features
from labeling import label_texts
from main import _json_safe, load_data, maybe_deduplicate, set_seed, split_data
from preprocessing import clean_texts
from train import _predict_proba, predict_with_threshold

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(Config.BASE_DIR, "models")
METRICS_PATH = os.path.join(Config.RESULTS_DIR, "metrics.json")
FIGURE_PATH = os.path.join(Config.RESULTS_DIR, "10_shap_top15.png")
SUMMARY_PATH = os.path.join(Config.RESULTS_DIR, "shap_summary.json")

N_BOOTSTRAP = 1000
# Stated explicitly (Section 7.2). For this feature count it equals what shap's
# current "auto" setting selects, so results do not change if that default changes.
L1_REG = "aic"
FIGURE_DPI = 600


# ====
# 1. ARTIFACTS AND CONSISTENCY
# ====

def load_artifacts(seed: int):
    """Load models/ and verify it matches metrics.json for the same seed."""
    files = {
        "tfidf": "tfidf_vectorizer.joblib",
        "pca": "pca_model.joblib",
        "scaler": "scaler.joblib",
        "model": "best_model.joblib",
    }
    missing = [
        f for f in list(files.values()) + ["model_info.json"]
        if not os.path.exists(os.path.join(MODELS_DIR, f))
    ]
    if missing:
        print(f"Error: missing artifact(s) {missing}. Run 'python save_model.py --seed {seed}' first.")
        sys.exit(1)
    if not os.path.exists(METRICS_PATH):
        print(f"Error: {METRICS_PATH} not found. Run 'python main.py --seed {seed}' first.")
        sys.exit(1)

    with open(os.path.join(MODELS_DIR, "model_info.json"), "r", encoding="utf-8") as file:
        info = json.load(file)
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        reference = json.load(file)

    problems = []
    if info.get("seed") != seed:
        problems.append(f"models/ was produced with seed={info.get('seed')}, not {seed}.")
    if reference.get("seed") != seed:
        problems.append(f"metrics.json was produced with seed={reference.get('seed')}, not {seed}.")
    if info.get("selected_model") != reference.get("selected_model"):
        problems.append(
            f"models/ holds '{info.get('selected_model')}', but metrics.json selected "
            f"'{reference.get('selected_model')}'."
        )
    if not info.get("consistent_with_metrics_json", False):
        problems.append("save_model.py reported that the persisted model does not reproduce metrics.json.")

    if problems:
        for p in problems:
            print(f"Error: {p}")
        print("Re-run 'python main.py' and 'python save_model.py' with the same seed.")
        sys.exit(1)

    artifacts = {k: joblib.load(os.path.join(MODELS_DIR, v)) for k, v in files.items()}
    return artifacts, info


def check_feature_layout(artifacts: dict, info: dict):
    """Feature order must match features.combine_features: TF-IDF, then BERT-PCA."""
    tfidf_names = list(artifacts["tfidf"].get_feature_names_out())
    n_tfidf = len(tfidf_names)
    n_pca = int(artifacts["pca"].n_components_)
    layout = info.get("feature_layout", {})

    expected = (layout.get("tfidf"), layout.get("bert_pca"))
    if expected != (n_tfidf, n_pca) or artifacts["scaler"].n_features_in_ != n_tfidf + n_pca:
        print(
            f"Error: feature layout mismatch (artifacts: {n_tfidf}+{n_pca}, "
            f"model_info: {expected}, scaler: {artifacts['scaler'].n_features_in_})."
        )
        sys.exit(1)

    names = tfidf_names + [f"bert_pca_{i}" for i in range(n_pca)]
    return names, n_tfidf, n_pca


def check_splits(info: dict, actual: dict) -> None:
    expected = info.get("splits", {})
    for key, value in actual.items():
        if key in expected and int(expected[key]) != int(value):
            print(
                f"Error: {key} split size {value:,} differs from models/model_info.json "
                f"({int(expected[key]):,}). Data or config changed since save_model.py was run."
            )
            sys.exit(1)


# ====
# 2. TEXT -> MODEL SPACE (transform only, persisted artifacts)
# ====

def to_model_space(bg_texts: list, ex_texts: list, artifacts: dict):
    """Apply persisted TF-IDF, PCA and scaler; BERT is the same frozen encoder."""
    tfidf = artifacts["tfidf"]
    X_bg_tfidf = tfidf.transform(clean_texts(bg_texts, method="tfidf")).toarray()
    X_ex_tfidf = tfidf.transform(clean_texts(ex_texts, method="tfidf")).toarray()

    X_bg_bert, X_ex_bert = create_bert_features(
        clean_texts(bg_texts, method="bert"),
        clean_texts(ex_texts, method="bert"),
        model_name=Config.BERT_MODEL,
        batch_size=Config.BATCH_SIZE,
        max_length=Config.MAX_LENGTH,
        use_fp16=Config.USE_FP16,
    )

    pca, scaler = artifacts["pca"], artifacts["scaler"]
    X_bg = combine_features(X_bg_tfidf, pca.transform(X_bg_bert))
    X_ex = combine_features(X_ex_tfidf, pca.transform(X_ex_bert))

    X_bg = scaler.transform(X_bg).astype(np.float32)
    X_ex = scaler.transform(X_ex).astype(np.float32)
    return X_bg, X_ex


# ====
# 3. STABILITY (Section 7.2)
# ====

def bootstrap_mean_abs(abs_shap: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap the explained reviews; returns (n_boot, n_features) mean |SHAP|."""
    rng = np.random.default_rng(seed)
    n, d = abs_shap.shape
    boot = np.empty((n_boot, d), dtype=np.float64)
    for b in range(n_boot):
        boot[b] = abs_shap[rng.integers(0, n, n)].mean(axis=0)
    return boot


def topn_frequency(boot: np.ndarray, top_n: int) -> np.ndarray:
    """Fraction of bootstrap replicates in which each feature is in the top N."""
    counts = np.zeros(boot.shape[1], dtype=np.float64)
    for row in boot:
        counts[np.argpartition(row, -top_n)[-top_n:]] += 1
    return counts / boot.shape[0]


# ====
# 4. MAIN
# ====

def main(seed: int | None = None) -> None:
    seed = Config.SEED if seed is None else seed

    print("\n" + "=" * 60)
    print(" SHAP ATTRIBUTION (shap.KernelExplainer)")
    print("=" * 60)
    print(f"Seed: {seed} | shap {shap.__version__}")

    set_seed(seed)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    artifacts, info = load_artifacts(seed)
    model_name = info["selected_model"]
    threshold = float(info["decision_threshold"])
    feature_names, n_tfidf, n_pca = check_feature_layout(artifacts, info)
    print(f"Model: {model_name} | frozen threshold (validation): {threshold:.4f}")
    print(f"Feature space: {n_tfidf} TF-IDF + {n_pca} BERT-PCA = {n_tfidf + n_pca}")

    # ---- Same data, same dedup, same labels, same split as main.py ----
    texts = load_data(Config.DATA_PATH, sample_size=Config.SAMPLE_SIZE, seed=seed)
    texts, _ = maybe_deduplicate(texts)
    labels, _ = label_texts(texts)
    train_texts, val_texts, test_texts, y_train, y_val, y_test = split_data(texts, labels, seed)
    check_splits(info, {"train": len(y_train), "validation": len(y_val), "test": len(y_test)})

    # ---- Background from train, explained reviews from test ----
    rng = np.random.default_rng(seed)
    bg_size = min(Config.SHAP_BACKGROUND_SIZE, len(train_texts))
    ex_size = min(Config.SHAP_EXPLAIN_SIZE, len(test_texts))
    bg_idx = rng.choice(len(train_texts), bg_size, replace=False)
    ex_idx = rng.choice(len(test_texts), ex_size, replace=False)

    bg_texts = [train_texts[i] for i in bg_idx]
    ex_texts = [test_texts[i] for i in ex_idx]
    y_ex = np.asarray(y_test)[ex_idx]

    del texts, labels, train_texts, val_texts, test_texts, y_train, y_val
    gc.collect()

    print(f"\nBackground: {bg_size} training reviews | Explained: {ex_size} test reviews")
    X_bg, X_ex = to_model_space(bg_texts, ex_texts, artifacts)

    model = artifacts["model"]
    proba_ex = _predict_proba(model, X_ex)
    if proba_ex is None:
        print(f"Error: '{model_name}' has no predict_proba; KernelExplainer needs probabilities.")
        sys.exit(1)

    # Threshold does not change SHAP values (they explain the spam probability);
    # it is used only to describe the explained sample.
    pred_ex = predict_with_threshold(proba_ex, threshold)
    print(
        f"Explained sample: true spam={int(y_ex.sum())}/{ex_size}, "
        f"predicted spam at frozen threshold={int(pred_ex.sum())}/{ex_size}"
    )

    # ---- Shapley values ----
    def predict_spam_proba(x):
        return _predict_proba(model, np.asarray(x, dtype=np.float32))

    print(f"\nRunning shap.KernelExplainer (nsamples='auto', l1_reg='{L1_REG}')...")
    print("This is the slowest step of the script.")
    explainer = shap.KernelExplainer(predict_spam_proba, X_bg)
    shap_values = np.asarray(
        explainer.shap_values(X_ex, nsamples="auto", l1_reg=L1_REG), dtype=np.float64
    )
    if shap_values.shape != X_ex.shape:
        print(f"Error: unexpected SHAP output shape {shap_values.shape}, expected {X_ex.shape}.")
        sys.exit(1)

    abs_shap = np.abs(shap_values)
    mean_abs = abs_shap.mean(axis=0)

    # ---- Stability: bootstrap CIs and top-N inclusion frequency ----
    top_n = Config.SHAP_TOP_N
    boot = bootstrap_mean_abs(abs_shap, N_BOOTSTRAP, seed)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5], axis=0)
    freq = topn_frequency(boot, top_n)

    # ---- Block contribution: TF-IDF vs BERT-PCA ----
    share_tfidf = float(mean_abs[:n_tfidf].sum() / mean_abs.sum())
    boot_share = boot[:, :n_tfidf].sum(axis=1) / boot.sum(axis=1)
    share_ci = np.percentile(boot_share, [2.5, 97.5])

    print(
        f"\nShare of total mean |SHAP|: TF-IDF {share_tfidf * 100:.1f}% "
        f"[{share_ci[0] * 100:.1f}, {share_ci[1] * 100:.1f}] | "
        f"BERT-PCA {(1 - share_tfidf) * 100:.1f}% "
        f"[{(1 - share_ci[1]) * 100:.1f}, {(1 - share_ci[0]) * 100:.1f}]"
    )

    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top = []
    print(f"\nTop {top_n} features by mean |SHAP value| (95% bootstrap CI, top-{top_n} frequency):")
    for rank, j in enumerate(top_idx, start=1):
        block = "tfidf" if j < n_tfidf else "bert_pca"
        top.append({
            "rank": rank,
            "feature": feature_names[j],
            "block": block,
            "mean_abs_shap": float(mean_abs[j]),
            "ci95": [float(ci_low[j]), float(ci_high[j])],
            "topn_frequency": float(freq[j]),
        })
        print(
            f"  {rank:>2}. {feature_names[j]:<28} {mean_abs[j]:.5f} "
            f"[{ci_low[j]:.5f}, {ci_high[j]:.5f}]  {freq[j] * 100:5.1f}%  ({block})"
        )

    # ---- Figure (600 dpi, grey-scale safe via hatching) ----
    print("\nGenerating plot...")
    order = top_idx[::-1]
    is_bert = order >= n_tfidf
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        np.arange(len(order)),
        mean_abs[order],
        xerr=[mean_abs[order] - ci_low[order], ci_high[order] - mean_abs[order]],
        color=["#DD8452" if b else "#4C72B0" for b in is_bert],
        edgecolor="black",
        capsize=3,
    )
    for bar, b in zip(bars, is_bert):
        if b:
            bar.set_hatch("//")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([feature_names[j] for j in order])
    ax.set_xlabel("Mean |SHAP value| (Shapley value on predicted spam probability)")
    ax.set_title(
        f"Top {top_n} features, shap.KernelExplainer, {model_name}\n"
        f"test n={ex_size}, train background n={bg_size}; error bars: 95% bootstrap CI\n"
        f"Total share: TF-IDF {share_tfidf * 100:.1f}% | BERT-PCA {(1 - share_tfidf) * 100:.1f}%",
        fontsize=11,
    )
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="#4C72B0", edgecolor="black"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#DD8452", edgecolor="black", hatch="//"),
        ],
        labels=["TF-IDF term", "BERT-PCA component"],
        loc="lower right",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- Summary (single source for the numbers quoted in Section IV-C) ----
    summary = {
        "method": "Shapley values (SHAP), not permutation importance",
        "explainer": "shap.KernelExplainer",
        "shap_version": shap.__version__,
        "nsamples": "auto",
        "nsamples_used": getattr(explainer, "nsamples", None),
        "l1_reg": L1_REG,
        "explained_output": "predicted spam probability",
        "model": model_name,
        "decision_threshold": threshold,
        "threshold_selected_on": info.get("threshold_selected_on", "validation"),
        "seed": seed,
        "background": {"source": "train split", "size": bg_size},
        "explained": {
            "source": "test split",
            "size": ex_size,
            "true_spam": int(y_ex.sum()),
            "predicted_spam_at_threshold": int(pred_ex.sum()),
        },
        "bootstrap_replicates": N_BOOTSTRAP,
        "feature_space": {"tfidf": n_tfidf, "bert_pca": n_pca, "total": n_tfidf + n_pca},
        "block_share_of_mean_abs_shap": {
            "tfidf": share_tfidf,
            "tfidf_ci95": [float(share_ci[0]), float(share_ci[1])],
            "bert_pca": 1 - share_tfidf,
            "bert_pca_ci95": [float(1 - share_ci[1]), float(1 - share_ci[0])],
        },
        "top_features": top,
        "figure": os.path.basename(FIGURE_PATH),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(_json_safe(summary), file, ensure_ascii=False, indent=2)

    print(f"\nSaved: {FIGURE_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print("=" * 60 + "\n")


# ====
# 5. ENTRY POINT
# ====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP attribution for the model selected by main.py")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Must match the seed used for main.py and save_model.py.",
    )
    args = parser.parse_args()
    main(seed=args.seed)
