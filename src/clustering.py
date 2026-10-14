"""Clustering track on the mfeat handwritten-digit dataset.

Rule followed throughout: the ground-truth digit labels are NEVER passed to a
fit. They are used only after clustering, in `external_validation` and
`map_clusters_to_digits`, to interpret the result.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

from . import config as C

RS = C.RANDOM_STATE


# ---------------------------------------------------------------- preprocessing
def scale(X):
    """Standardise: the 6 feature sets live on very different scales
    (pixel counts 0-6 vs. Karhunen-Loeve coefficients ~ +/-10 vs. morphology in the 100s)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# ---------------------------------------------------------------- K-Means
def elbow_analysis(X, k_range=C.K_RANGE):
    """Inertia (within-cluster sum of squares) and silhouette for each k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def find_elbow(ks, inertias):
    """Kneedle-style elbow: the point farthest from the straight line that
    joins the first and last points of the normalised inertia curve."""
    x = np.asarray(ks, float)
    y = np.asarray(inertias, float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    # distance from each point to the line from (0,1) to (1,0)
    dist = np.abs(xn + yn - 1) / np.sqrt(2)
    return int(x[np.argmax(dist)])


def fit_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=20, random_state=RS).fit(X)


# ---------------------------------------------------------------- Agglomerative
def fit_agglomerative(X, k, linkage="ward"):
    return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(X)


def compare_linkages(X, k, linkages=C.LINKAGES):
    """Fit one Agglomerative model per linkage; return (metrics table, fitted models)."""
    rows, models = [], {}
    for lk in linkages:
        m = fit_agglomerative(X, k, lk)
        models[lk] = m
        sizes = np.bincount(m.labels_)
        rows.append({"Linkage": lk, **internal_metrics(X, m.labels_),
                     "Largest cluster": int(sizes.max()),
                     "Smallest cluster": int(sizes.min())})
    return pd.DataFrame(rows), models


def linkage_matrix(X, method="ward"):
    """SciPy linkage matrix used to draw the dendrogram."""
    return scipy_linkage(X, method=method)


def centroids_from_labels(X, labels):
    """Agglomerative clustering has no predict(); we assign new points to the
    nearest cluster mean, which is what the UI uses."""
    return np.vstack([X[labels == c].mean(axis=0) for c in np.unique(labels)])


def nearest_centroid(X_new, centroids):
    d = ((X_new[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


# ---------------------------------------------------------------- evaluation
def internal_metrics(X, labels):
    """The three mandatory label-free metrics.
    Silhouette: higher is better (-1..1). Davies-Bouldin: lower is better (>=0).
    Calinski-Harabasz: higher is better."""
    return {
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
    }


def external_validation(y_true, labels):
    """Post-hoc only: agreement between clusters and real digits."""
    return {"ARI": adjusted_rand_score(y_true, labels),
            "NMI": normalized_mutual_info_score(y_true, labels)}


def map_clusters_to_digits(y_true, labels):
    """Post-hoc: majority digit in each cluster and its purity."""
    ct = pd.crosstab(pd.Series(labels, name="Cluster"), pd.Series(y_true, name="Digit"))
    return pd.DataFrame({"Majority digit": ct.idxmax(axis=1),
                         "Purity": (ct.max(axis=1) / ct.sum(axis=1)).round(3),
                         "Size": ct.sum(axis=1)}), ct


# ---------------------------------------------------------------- projections
def pca_2d(X):
    pca = PCA(n_components=2, random_state=RS)
    return pca.fit_transform(X), pca


def tsne_2d(X, perplexity=30):
    # PCA to 50 dims first is the standard recipe: faster and less noisy t-SNE.
    X50 = PCA(n_components=50, random_state=RS).fit_transform(X)
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=RS).fit_transform(X50)
