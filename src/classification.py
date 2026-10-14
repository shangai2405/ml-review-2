"""Classification track: the 10 algorithms from guideline section 3.2
(Part A = Review 1, Part B = Review 2), evaluation, and hyperparameter tuning."""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C

RS = C.RANDOM_STATE
METRICS = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)", "ROC-AUC (OvR)"]


def _pipe(model):
    """Every model gets the same scaler inside a Pipeline, so the scaler is fitted on the
    training data only (also inside each CV fold) - no leakage."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def part_a_models():
    """Review 1 Part A: Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVC."""
    return {
        "Logistic Regression": _pipe(LogisticRegression(max_iter=5000, random_state=RS)),
        "KNN": _pipe(KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes": _pipe(GaussianNB()),
        "Decision Tree": _pipe(DecisionTreeClassifier(random_state=RS)),
        "SVM (SVC)": _pipe(SVC(probability=True, random_state=RS)),
    }


def part_b_models():
    """Review 2 Part B: Random Forest, AdaBoost, Gradient Boosting (sklearn GBM),
    Bagging (Decision Tree base estimator), MLP."""
    return {
        "Random Forest": _pipe(RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=-1)),
        "AdaBoost": _pipe(AdaBoostClassifier(n_estimators=50, random_state=RS)),
        "Gradient Boosting": _pipe(GradientBoostingClassifier(random_state=RS)),
        "Bagging": _pipe(BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS),
                                           n_estimators=100, random_state=RS, n_jobs=-1)),
        "MLP (Neural Net)": _pipe(MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                                                random_state=RS)),
    }


def all_models():
    return {**part_a_models(), **part_b_models()}


def part_of(name):
    return "A (Review 1)" if name in part_a_models() else "B (Review 2)"


def evaluate(model, X_test, y_test):
    """Mandatory metrics: Accuracy, weighted Precision/Recall/F1, OvR ROC-AUC."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    return dict(zip(METRICS, [
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, average="weighted"),
        recall_score(y_test, pred, average="weighted"),
        f1_score(y_test, pred, average="weighted"),
        roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"),
    ]))


def confusion(model, X_test, y_test):
    return confusion_matrix(y_test, model.predict(X_test))


def train_and_compare(X_train, X_test, y_train, y_test):
    """Fit all 10 models with default-ish settings on the same split.
    Returns (comparison DataFrame ranked by weighted F1, dict of fitted models)."""
    rows, fitted = [], {}
    for name, model in all_models().items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fitted[name] = model
        rows.append({"Model": name, "Part": part_of(name), **evaluate(model, X_test, y_test),
                     "Train time (s)": round(time.perf_counter() - t0, 3)})
    table = (pd.DataFrame(rows).sort_values(["F1 (weighted)", "Accuracy"], ascending=False)
             .reset_index(drop=True))
    table.index += 1
    return table, fitted


# Search spaces follow the "Notes" column of guideline section 3.2.
PARAM_GRIDS = {
    # Part A
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100, 1000]},
    "KNN": {"model__n_neighbors": [3, 5, 7, 11, 15, 21, 31],
            "model__metric": ["euclidean", "manhattan"]},                 # "discuss distance metrics"
    "Naive Bayes": {"model__var_smoothing": np.logspace(-11, -5, 7)},
    "Decision Tree": {"model__max_depth": [3, 5, 8, 12, None],
                      "model__min_samples_leaf": [1, 5, 10]},
    "SVM (SVC)": {"model__C": [0.1, 1, 10, 100], "model__kernel": ["linear", "rbf"]},
    # Part B
    "Random Forest": {"model__n_estimators": [100, 300, 600], "model__max_depth": [None, 10, 20]},
    "AdaBoost": {"model__n_estimators": [50, 100, 200, 400], "model__learning_rate": [0.1, 0.5, 1.0],
                 "model__estimator": [DecisionTreeClassifier(max_depth=d, random_state=RS)
                                      for d in (1, 2, 3)]},
    "Gradient Boosting": {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1],
                          "model__max_depth": [2, 3]},
    "Bagging": {"model__n_estimators": [50, 100, 200], "model__max_samples": [0.5, 1.0],
                "model__max_features": [0.5, 1.0]},
    "MLP (Neural Net)": {"model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
                         "model__activation": ["relu", "tanh"]},
}


def tune(name, X_train, y_train):
    """5-fold stratified GridSearchCV on weighted F1 (the CV requirement of guideline 7.2)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    gs = GridSearchCV(all_models()[name], PARAM_GRIDS[name], scoring="f1_weighted", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs


def readable_params(params):
    """{'model__C': 100} -> 'C=100' ; estimators shown by their max_depth."""
    out = []
    for k, v in params.items():
        k = k.replace("model__", "")
        if hasattr(v, "max_depth"):
            v = f"DecisionTree(max_depth={v.max_depth})"
        out.append(f"{k}={v}")
    return ", ".join(out)


def save_model(model, feature_names, name, metrics, path=None):
    path = path or C.MODELS_DIR / "best_classifier.joblib"
    joblib.dump({"model": model, "features": list(feature_names), "name": name,
                 "metrics": metrics}, path)
    return path
