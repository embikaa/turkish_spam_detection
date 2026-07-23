"""Visualization module for spam detection analysis and reporting."""
import os
import glob
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

from preprocessing import STOPWORDS

COLORS = {
    "bg": "white",
    "grid": "#E5E5E5",
    "text": "black",
    "tick": "black",
    "spam": "#D62728",
    "genuine": "#1F77B4",
    "accent": "#FF7F0E",
}


def setup_style() -> None:
    """Configure matplotlib styling for academic-quality figures."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": "black",
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["tick"],
        "ytick.color": COLORS["tick"],
        "grid.color": COLORS["grid"],
        "font.family": "sans-serif",
        "font.size": 12,
    })


def clean_old_plots(output_dir: str) -> None:
    """
    Remove existing PNG files from output directory.

    Args:
        output_dir: Directory path to clean.
    """
    if os.path.exists(output_dir):
        files = glob.glob(os.path.join(output_dir, "*.png"))
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        if files:
            print(f"  Removed {len(files)} old plots.")


def plot_system_architecture(output_dir: str) -> None:
    """
    Generate system architecture flowchart.

    Args:
        output_dir: Output directory for the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(1, 5)
    ax.axis('off')

    boxes = [
        (1, 2.5, 'Raw Data\n(CSV)'),
        (3.5, 2.5, 'Preprocessing\n(Cleaning)'),
        (6, 2.5, 'Feature\nExtraction\n(Hybrid)'),
        (9, 2.5, 'Classification\nModels'),
        (11.5, 2.5, 'Performance\nEvaluation'),
    ]

    for x, y, text in boxes:
        fancy_box = matplotlib.patches.FancyBboxPatch(
            (x, y), 2, 1.2, boxstyle="round, pad=0.1",
            facecolor='white', edgecolor='black', linewidth=1.5
        )
        ax.add_patch(fancy_box)
        ax.text(
            x + 1, y + 0.6, text, ha='center', va='center',
            fontsize=11, color='black', fontweight='bold'
        )

    for i in range(len(boxes) - 1):
        start = (boxes[i][0] + 2, boxes[i][1] + 0.6)
        end = (boxes[i + 1][0], boxes[i + 1][1] + 0.6)
        ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(arrowstyle='->', color='black', lw=2.0)
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "0_system_architecture.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("  System architecture plot saved.")


def plot_label_distribution(label_stats: Dict, output_dir: str) -> None:
    """
    Generate pie chart showing spam/genuine label distribution.

    Args:
        label_stats: Statistics from weak labeling.
        output_dir: Output directory for the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [label_stats['spam'], label_stats['genuine']]
    labels = [
        f"Spam\n({label_stats['spam_pct']}%)",
        f"Genuine\n({label_stats['genuine_pct']}%)"
    ]
    colors = [COLORS["spam"], COLORS["genuine"]]
    explode = (0.05, 0)

    ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='', shadow=False, startangle=90,
        textprops={'fontsize': 13, 'color': 'black', 'fontweight': 'bold'},
        wedgeprops={'linewidth': 1, 'edgecolor': 'black'}
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "1_label_distribution.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("  Label distribution plot saved.")


def plot_oversampling_effect(
    y_train_orig: np.ndarray,
    y_train_res: np.ndarray,
    output_dir: str
) -> None:
    """
    Generate bar charts comparing class distribution before/after oversampling.

    Args:
        y_train_orig: Original training labels.
        y_train_res: Resampled training labels.
        output_dir: Output directory for the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    _, counts = np.unique(y_train_orig, return_counts=True)
    axes[0].bar(
        ['Genuine', 'Spam'], counts,
        color=[COLORS["genuine"], COLORS["spam"]], edgecolor='black'
    )
    axes[0].set_title('(a) Before Oversampling', fontsize=12)
    axes[0].set_ylabel('Number of Samples')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    _, counts_res = np.unique(y_train_res, return_counts=True)
    axes[1].bar(
        ['Genuine', 'Spam'], counts_res,
        color=[COLORS["genuine"], COLORS["spam"]], edgecolor='black'
    )
    axes[1].set_title('(b) After Oversampling', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    for ax in axes:
        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5),
                textcoords='offset points', fontsize=11
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "2_oversampling_effect.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("  Oversampling plot saved.")


def plot_best_model_cm(results: Dict, output_dir: str) -> None:
    best_name, best_metrics = sorted(
        results.items(), key=lambda x: x[1].get("f1", 0), reverse=True
    )[0]

    if "cm" not in best_metrics:
        return

    print(f"  Plotting confusion matrix for {best_name}...")
    cm = np.array(best_metrics["cm"])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Genuine', 'Spam'],
        yticklabels=['Genuine', 'Spam'],
        annot_kws={"size": 14}, cbar=False,
        linewidths=0.5, linecolor='black'
    )
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "3_best_model_cm.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("  Confusion matrix saved.")


def plot_top3_comparison(results: Dict, output_dir: str) -> None:
    """
    Generate grouped bar chart comparing top 3 models by F1 score.

    Args:
        results: Model evaluation results.
        output_dir: Output directory for the plot.
    """
    top3 = sorted(
        results.items(), key=lambda x: x[1].get("f1", 0), reverse=True
    )[:3]
    names = [x[0] for x in top3]
    metrics = {
        'Precision': [x[1].get('precision', 0) for x in top3],
        'Recall': [x[1].get('recall', 0) for x in top3],
        'F1-Score': [x[1].get('f1', 0) for x in top3],
    }

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    rects1 = ax.bar(
        x - width, metrics['Precision'], width,
        label='Precision', color=COLORS["genuine"], edgecolor='black'
    )
    rects2 = ax.bar(
        x, metrics['Recall'], width,
        label='Recall', color=COLORS["accent"], edgecolor='black'
    )
    rects3 = ax.bar(
        x + width, metrics['F1-Score'], width,
        label='F1-Score', color=COLORS["spam"], edgecolor='black'
    )

    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "4_top3_comparison.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("  Model comparison plot saved.")


def plot_auc_curve(results: Dict, output_dir: str) -> None:
    best_name, best_metrics = sorted(
        results.items(), key=lambda x: x[1].get("f1", 0), reverse=True
    )[0]

    if "fpr" not in best_metrics:
        return

    fpr = best_metrics["fpr"]
    tpr = best_metrics["tpr"]
    roc_auc = best_metrics.get("auc", 0)

    print(f"  Plotting AUC curve for {best_name}...")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr, tpr, color=COLORS["spam"], lw=2,
        label=f'{best_name} (AUC = {roc_auc:.2f})'
    )
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "5_auc_curve.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("  AUC curve saved.")


def plot_wordclouds(
    texts: List[str],
    labels: np.ndarray,
    output_dir: str
) -> None:
    """
    Generate word clouds for spam and genuine text samples.

    Args:
        texts: List of all text samples.
        labels: Corresponding labels (0=genuine, 1=spam).
        output_dir: Output directory for the plot.
    """
    if not WORDCLOUD_AVAILABLE:
        print("  WordCloud not installed, skipping.")
        return

    print("  Generating word clouds...")

    spam_texts = " ".join(t for t, l in zip(texts, labels) if l == 1)
    genuine_texts = " ".join(t for t, l in zip(texts, labels) if l == 0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    if len(spam_texts) > 0:
        wc_spam = WordCloud(
            width=800, height=400, background_color='white',
            colormap="Reds", stopwords=STOPWORDS
        ).generate(spam_texts)
        axes[0].imshow(wc_spam, interpolation='bilinear')
        axes[0].set_title('(a) Spam Reviews', fontsize=14, fontweight='bold')
        axes[0].axis('off')

    if len(genuine_texts) > 0:
        wc_genuine = WordCloud(
            width=800, height=400, background_color='white',
            colormap="Blues", stopwords=STOPWORDS
        ).generate(genuine_texts)
        axes[1].imshow(wc_genuine, interpolation='bilinear')
        axes[1].set_title('(b) Genuine Reviews', fontsize=14, fontweight='bold')
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "6_wordclouds.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("  Word clouds saved.")


def plot_tsne_distribution(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    sample_size: int = 1500,
    seed: int = 42
) -> None:
    """
    Generate t-SNE visualization of feature space distribution.

    Args:
        X: Feature matrix.
        y: Labels.
        output_dir: Output directory for the plot.
        sample_size: Maximum samples for t-SNE computation.
        seed: Random seed for reproducibility.
    """
    print("  Computing t-SNE visualization...")

    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
        y_sample = y[idx]
    else:
        X_sample = X
        y_sample = y

    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    X_tsne = tsne.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [COLORS["genuine"], COLORS["spam"]]
    scatter = sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_sample,
        palette=colors, alpha=0.7, edgecolor="k", s=50, ax=ax
    )

    handles, _ = scatter.get_legend_handles_labels()
    ax.legend(handles, ['Genuine (0)', 'Spam (1)'], title="Classes", loc="best")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "7_tsne_plot.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("  t-SNE plot saved.")


def plot_pca_variance(pca_model: Any, output_dir: str) -> None:
    """
    Generate PCA explained variance cumulative curve.

    Args:
        pca_model: Fitted PCA model.
        output_dir: Output directory for the plot.
    """
    print("  Generating PCA variance plot...")

    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        range(1, len(cumulative_variance) + 1), cumulative_variance,
        color=COLORS["accent"], linewidth=2, label="Cumulative Variance"
    )
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")

    n = pca_model.n_components_
    ax.axvline(
        x=n, color=COLORS["spam"], linestyle="--", alpha=0.7,
        label=f"Selected Dimension ({n})"
    )
    ax.axhline(
        y=cumulative_variance[n - 1], color=COLORS["spam"],
        linestyle="--", alpha=0.7
    )
    ax.scatter(
        n, cumulative_variance[n - 1], color=COLORS["spam"],
        s=100, zorder=5
    )
    ax.annotate(
        f'{cumulative_variance[n - 1] * 100:.1f}%',
        (n, cumulative_variance[n - 1]),
        textcoords="offset points", xytext=(-15, 10),
        ha='center', fontsize=11, fontweight="bold"
    )

    ax.grid(linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "8_pca_variance.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("  PCA variance plot saved.")


def plot_feature_importance(
    results: Dict,
    tfidf_vectorizer: Any,
    output_dir: str,
    top_n: int = 15
) -> None:
    rf_metrics = results.get("Random Forest", {})
    if "feature_importances" not in rf_metrics:
        return

    print("  Plotting feature importance...")

    importances = rf_metrics["feature_importances"]
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_importances = importances[:len(feature_names)]

    indices = np.argsort(tfidf_importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = tfidf_importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        x=top_scores, y=top_features,
        palette="Reds_r", edgecolor="black", ax=ax
    )

    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("TF-IDF Terms")
    ax.grid(axis='x', linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "9_feature_importance.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("  Feature importance plot saved.")


def create_all_plots(
    results: Dict,
    label_stats: Dict,
    output_dir: str,
    texts: Optional[List[str]] = None,
    labels: Optional[np.ndarray] = None,
    X_train: Optional[np.ndarray] = None,
    y_train_orig: Optional[np.ndarray] = None,
    y_train_res: Optional[np.ndarray] = None,
    pca_model: Optional[Any] = None,
    tfidf_vectorizer: Optional[Any] = None,
    seed: int = 42
) -> None:
    setup_style()
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating visualization plots...")
    clean_old_plots(output_dir)

    plot_system_architecture(output_dir)

    if label_stats:
        plot_label_distribution(label_stats, output_dir)

    if y_train_orig is not None and y_train_res is not None:
        plot_oversampling_effect(y_train_orig, y_train_res, output_dir)

    if results:
        plot_best_model_cm(results, output_dir)
        plot_top3_comparison(results, output_dir)
        plot_auc_curve(results, output_dir)

    if texts is not None and labels is not None:
        plot_wordclouds(texts, labels, output_dir)

    if X_train is not None and y_train_res is not None:
        plot_tsne_distribution(X_train, y_train_res, output_dir, seed=seed)

    if pca_model is not None:
        plot_pca_variance(pca_model, output_dir)

    if tfidf_vectorizer is not None:
        plot_feature_importance(results, tfidf_vectorizer, output_dir)

    print(f"\nAll plots saved to '{output_dir}'.")
