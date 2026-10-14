"""End-to-end pipeline for Review 2.

    python train_pipeline.py                 # both tracks
    python train_pipeline.py --only clf      # classification only
    python train_pipeline.py --only clu      # clustering only
    python train_pipeline.py --no-tsne       # skip t-SNE (faster)

Outputs
    models/best_classifier.joblib     tuned best classifier (used by the app)
    models/clustering.joblib          scaler, K-Means, Agglomerative centroids, projections
    reports/*.csv, reports/results.json
    reports/figures/*.png
"""
import argparse
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src import classification as clf
from src import clustering as clu
from src import config as C
from src import data_loader as dl
from src import visualize as viz


def _save(fig, name):
    if fig is not None:
        fig.savefig(C.FIG_DIR / name, dpi=130, bbox_inches="tight")
        plt.close(fig)


# =================================================================== classification
def run_classification():
    print("\n=== CLASSIFICATION (Mobile Price, 4 classes) ===")
    X_train, X_test, y_train, y_test = dl.split_classification()
    print(f"train {X_train.shape}, test {X_test.shape}")
    labels = [C.PRICE_LABELS[i] for i in range(4)]

    # 1. Baseline: all 10 models on the same split
    table, fitted = clf.train_and_compare(X_train, X_test, y_train, y_test)
    table.to_csv(C.REPORTS_DIR / "classification_comparison.csv", index_label="Rank")
    print(table.round(4).to_string())
    _save(viz.classifier_comparison(table, clf.METRICS), "clf_comparison.png")
    _save(viz.confusion_grid({n: clf.confusion(m, X_test, y_test) for n, m in fitted.items()},
                             labels, "Confusion matrices - all 10 models (baseline)"),
          "clf_confusion_all.png")

    # 2. Tune every model with 5-fold GridSearchCV (grids follow guideline 3.2 notes)
    tuning_rows, tuned = [], {}
    for name in table["Model"]:
        print(f"tuning {name} ...")
        gs = clf.tune(name, X_train, y_train)
        tuned[name] = gs
        after = clf.evaluate(gs.best_estimator_, X_test, y_test)
        before = table.set_index("Model").loc[name]
        tuning_rows.append({
            "Model": name, "Part": clf.part_of(name), "Best params": clf.readable_params(gs.best_params_),
            "CV F1 (5-fold)": gs.best_score_, "CV F1 std": gs.cv_results_["std_test_score"][gs.best_index_],
            "Accuracy before": before["Accuracy"], "Accuracy after": after["Accuracy"],
            "F1 before": before["F1 (weighted)"], "F1 after": after["F1 (weighted)"],
            "ROC-AUC before": before["ROC-AUC (OvR)"], "ROC-AUC after": after["ROC-AUC (OvR)"],
        })
    tuning = (pd.DataFrame(tuning_rows).sort_values("CV F1 (5-fold)", ascending=False)
              .reset_index(drop=True))
    tuning.index += 1
    tuning.to_csv(C.REPORTS_DIR / "classification_tuning.csv", index_label="Rank")
    print(tuning.drop(columns="Best params").round(4).to_string())

    # Final consolidated table: all 10 tuned models, all mandatory metrics
    final = pd.DataFrame([{"Model": n, "Part": clf.part_of(n),
                           **clf.evaluate(gs.best_estimator_, X_test, y_test)} for n, gs in tuned.items()])
    final = final.sort_values(["F1 (weighted)", "Accuracy"], ascending=False).reset_index(drop=True)
    final.index += 1
    final.to_csv(C.REPORTS_DIR / "classification_final.csv", index_label="Rank")
    _save(viz.confusion_grid({n: clf.confusion(gs.best_estimator_, X_test, y_test) for n, gs in tuned.items()},
                             labels, "Confusion matrices - all 10 models (tuned)"), "clf_confusion_all_tuned.png")

    # 3. Final pick by cross-validated F1 (the test set is not used to choose)
    best_name = tuning["Model"].iloc[0]
    best_model = tuned[best_name].best_estimator_
    best_metrics = clf.evaluate(best_model, X_test, y_test)
    clf.save_model(best_model, X_train.columns, best_name, best_metrics)
    print(f"BEST: {best_name} -> {best_metrics}")
    _save(viz.confusion(best_model, X_test, y_test, f"{best_name} (tuned) - confusion matrix", labels),
          "clf_best_confusion.png")
    _save(viz.feature_importance(tuned["Random Forest"].best_estimator_, X_train.columns,
                                 "Random Forest (tuned) - feature importance"), "clf_rf_importance.png")
    _save(viz.feature_importance(best_model, X_train.columns,
                                 f"{best_name} (tuned) - mean |coefficient|"), "clf_best_importance.png")
    return {"comparison": table.round(4).to_dict("records"),
            "tuning": tuning.round(4).to_dict("records"),
            "final": final.round(4).to_dict("records"),
            "best_model": best_name, "best_params": clf.readable_params(tuned[best_name].best_params_),
            "best_metrics": best_metrics}


# =================================================================== clustering
def run_clustering(do_tsne=True):
    print("\n=== CLUSTERING (mfeat digits, 649 features) ===")
    X_raw, y, _ = dl.load_mfeat()
    X, scaler = clu.scale(X_raw)              # labels y are NOT used below until "post-hoc"

    # 1. Elbow method for K-Means
    elbow = clu.elbow_analysis(X)
    elbow.to_csv(C.REPORTS_DIR / "kmeans_elbow.csv", index=False)
    k_elbow = clu.find_elbow(elbow["k"], elbow["inertia"])
    k_sil = int(elbow.loc[elbow["silhouette"].idxmax(), "k"])
    k = C.K_FINAL or k_elbow
    print(f"automatic knee k = {k_elbow}, best-silhouette k = {k_sil} -> using k = {k}")
    _save(viz.elbow_plot(elbow, k, k_auto=k_elbow), "clu_elbow.png")

    # Sensitivity: how do the metrics move for the candidate k values?
    sens = []
    for kk in sorted(set(C.K_SENSITIVITY + [k_elbow, k_sil, k])):
        lab_km = clu.fit_kmeans(X, kk).labels_
        lab_ag = clu.fit_agglomerative(X, kk, "ward").labels_
        for algo, lab in (("K-Means", lab_km), ("Agglomerative (ward)", lab_ag)):
            sens.append({"k": kk, "Algorithm": algo, **clu.internal_metrics(X, lab),
                         **clu.external_validation(y, lab)})   # ARI/NMI: post-hoc only
    sens = pd.DataFrame(sens)
    sens.to_csv(C.REPORTS_DIR / "k_sensitivity.csv", index=False)
    print(sens.round(4).to_string(index=False))

    # 2. K-Means
    km = clu.fit_kmeans(X, k)

    # 3. Agglomerative: compare linkages, draw dendrogram
    link_table, agg_models = clu.compare_linkages(X, k)
    link_table.to_csv(C.REPORTS_DIR / "agglomerative_linkages.csv", index=False)
    print(link_table.round(4).to_string(index=False))
    for method in ["ward", "complete", "average"]:
        _save(viz.dendrogram_plot(clu.linkage_matrix(X, method), k, method),
              f"clu_dendrogram_{method}.png")
    best_link = "ward"   # see README: highest Silhouette/CH among balanced linkages
    agg = agg_models[best_link]

    # 4. Internal metrics for every algorithm
    all_labels = {"K-Means": km.labels_,
                  **{f"Agglomerative ({lk})": m.labels_ for lk, m in agg_models.items()}}
    metrics = pd.DataFrame([{"Algorithm": n, **clu.internal_metrics(X, lab)}
                            for n, lab in all_labels.items()])

    # 5. Post-hoc validation against true digits (never used in fitting)
    ext = pd.DataFrame([{"Algorithm": n, **clu.external_validation(y, lab)}
                        for n, lab in all_labels.items()])
    metrics = metrics.merge(ext, on="Algorithm")
    metrics.to_csv(C.REPORTS_DIR / "clustering_metrics.csv", index=False)
    print(metrics.round(4).to_string(index=False))
    _save(viz.metric_bars(metrics, "Algorithm"), "clu_metric_bars.png")

    for name, lab in [("K-Means", km.labels_), (f"Agglomerative ({best_link})", agg.labels_)]:
        mapping, ct = clu.map_clusters_to_digits(y, lab)
        slug = "kmeans" if name == "K-Means" else "agglomerative"
        mapping.to_csv(C.REPORTS_DIR / f"cluster_digit_map_{slug}.csv")
        _save(viz.contingency_heatmap(ct, f"{name}: cluster vs. true digit (post-hoc)"),
              f"clu_contingency_{slug}.png")

    # 6. Visualisation: PCA (mandatory) and t-SNE (encouraged)
    pca_xy, pca = clu.pca_2d(X)
    print(f"PCA 2D explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    _save(viz.scatter_grid(pca_xy, all_labels, "PCA", "PCA (2 components) - every algorithm"),
          "clu_pca_all.png")
    _save(viz.scatter_clusters(pca_xy, y, "Ground-truth digits (reference only)"),
          "clu_pca_truth.png")
    tsne_xy = None
    if do_tsne:
        print("running t-SNE ...")
        tsne_xy = clu.tsne_2d(X)
        _save(viz.scatter_grid(tsne_xy, {"K-Means": km.labels_,
                                         f"Agglomerative ({best_link})": agg.labels_,
                                         "Ground truth (reference)": y}, "t-SNE",
                               "t-SNE projection"), "clu_tsne.png")

    # 7. Persist everything the app needs
    joblib.dump({
        "scaler": scaler, "kmeans": km, "k": k, "k_silhouette": k_sil,
        "agg_labels": {lk: m.labels_ for lk, m in agg_models.items()},
        "agg_centroids": {lk: clu.centroids_from_labels(X, m.labels_)
                          for lk, m in agg_models.items()},
        "best_linkage": best_link, "pca": pca, "pca_xy": pca_xy, "tsne_xy": tsne_xy,
        "elbow": elbow, "metrics": metrics, "linkage_table": link_table, "y_true": y,
    }, C.MODELS_DIR / "clustering.joblib")
    return {"k_auto_knee": k_elbow, "k_best_silhouette": k_sil, "k_used": k,
            "k_sensitivity": sens.round(4).to_dict("records"),
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "metrics": metrics.round(4).to_dict("records"),
            "linkages": link_table.round(4).to_dict("records")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["clf", "clu"])
    ap.add_argument("--no-tsne", action="store_true")
    a = ap.parse_args()
    out_path = C.REPORTS_DIR / "results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    if a.only in (None, "clf"):
        results["classification"] = run_classification()
    if a.only in (None, "clu"):
        results["clustering"] = run_clustering(do_tsne=not a.no_tsne)
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved results to {out_path}")
