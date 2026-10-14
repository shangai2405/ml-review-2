"""All figures. Every function returns a matplotlib Figure so the same code is
used by the pipeline (saved to reports/figures), the notebooks and the Gradio app."""
import sys

import matplotlib

if "ipykernel" not in sys.modules:   # headless backend for the pipeline/app, inline in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import ConfusionMatrixDisplay

CMAP = "tab10"


def elbow_plot(elbow_df, chosen_k, k_auto=None):
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(elbow_df["k"], elbow_df["inertia"], "o-", color="#1f4e79", label="Inertia")
    if k_auto is not None and k_auto != chosen_k:
        ax1.axvline(k_auto, ls=":", color="grey", label=f"Automatic knee k = {k_auto}")
    ax1.axvline(chosen_k, ls="--", color="#c0392b", label=f"Chosen k = {chosen_k}")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Inertia (within-cluster SSE)")
    ax2 = ax1.twinx()
    ax2.plot(elbow_df["k"], elbow_df["silhouette"], "s--", color="#e67e22", alpha=.8,
             label="Silhouette")
    ax2.set_ylabel("Silhouette score")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title("K-Means elbow curve (inertia vs. k)")
    fig.tight_layout()
    return fig


def dendrogram_plot(Z, k, method="ward", truncate_p=40):
    fig, ax = plt.subplots(figsize=(11, 5))
    # Horizontal cut that yields exactly k clusters
    cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2
    dendrogram(Z, truncate_mode="lastp", p=truncate_p, color_threshold=cut,
               show_leaf_counts=True, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.axhline(cut, ls="--", color="#c0392b", label=f"Cut for k = {k}")
    ax.set_title(f"Hierarchical clustering dendrogram ({method} linkage, last {truncate_p} merges)")
    ax.set_xlabel("Cluster (sample count in brackets)")
    ax.set_ylabel("Merge distance")
    ax.legend()
    fig.tight_layout()
    return fig


def scatter_clusters(coords, labels, title, method="PCA", ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=CMAP, s=8, alpha=.75)
    ax.set_title(title)
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.legend(*sc.legend_elements(), title="Cluster", fontsize=7, title_fontsize=8,
              loc="best", markerscale=.8, ncol=2)
    if own:
        fig.tight_layout()
        return fig


def scatter_grid(coords, label_sets, method="PCA", suptitle=None):
    n = len(label_sets)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, labels) in zip(axes, label_sets.items()):
        scatter_clusters(coords, labels, title, method, ax=ax)
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


def metric_bars(df, name_col):
    """Side-by-side bars for the three internal metrics."""
    metrics = [("Silhouette", "higher is better"), ("Davies-Bouldin", "lower is better"),
               ("Calinski-Harabasz", "higher is better")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (m, hint) in zip(axes, metrics):
        ax.bar(df[name_col], df[m], color="#2e86c1")
        ax.set_title(f"{m} ({hint})")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(m)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(df[m]):
            ax.text(i, v, f"{v:.3f}" if v < 100 else f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def contingency_heatmap(ct, title):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def digit_image(img, title=""):
    fig, ax = plt.subplots(figsize=(2.6, 2.8))
    ax.imshow(img, cmap="gray_r", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def classifier_comparison(table, metrics):
    t = table.set_index("Model")[metrics]
    fig, ax = plt.subplots(figsize=(13, 5))
    t.plot(kind="bar", ax=ax, width=.8, colormap="tab10")
    ax.set_ylim(max(0, t.values.min() - .05), 1.0)
    ax.set_title("All 10 classifiers on the same test split")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Metric", loc="lower left", ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def confusion_grid(matrices, labels, suptitle):
    """One confusion matrix per model (dict name -> 2-D array)."""
    import seaborn as sns
    n = len(matrices)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows))
    for ax, (name, cm) in zip(axes.ravel(), matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


def confusion(model, X_test, y_test, title, labels=None):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=labels,
                                          cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def feature_importance(model, feature_names, title, top=12):
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_).mean(axis=0)
    else:
        return None
    order = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(feature_names)[order][::-1], imp[order][::-1], color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig
