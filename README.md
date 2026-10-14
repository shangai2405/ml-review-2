---
title: ML Review 2 - Classification and Clustering
emoji: 📊
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app/app.py
pinned: false
---

# ML Capstone, Review 2: Classification Part B and Clustering

23CSE301 Machine Learning capstone. This repository holds the Review 2 deliverables: Classification Part B with the consolidated 10-algorithm comparison, the full Clustering track, an end-to-end pipeline, and an interactive Gradio app. The Review 1 notebook is also included as `notebooks/regression.ipynb`.

| Track | Dataset | Task |
|---|---|---|
| Classification | Mobile Price Classification (`data/train.csv`) | Predict a phone's price band (4 classes) |
| Clustering | UCI Multiple Features digits (`data/mfeat/`) | Group 2,000 handwritten digits **without labels** |

## 1. Datasets

**Mobile Price Classification** has 2,000 phones with 20 numeric features (battery, RAM, cameras, screen size and resolution, connectivity flags, and others). The target is `price_range` with four balanced classes (500 each): 0 = low cost, 1 = medium, 2 = high, 3 = very high. There are no missing values.

**Multiple Features (mfeat)** comes from the UCI repository and was donated by R. P. W. Duin (TU Delft). It contains 2,000 handwritten digits 0-9 (200 per digit) from Dutch utility maps. Each digit is described by six feature files with matching rows:

| File | Features | Description |
|---|---|---|
| `mfeat-fou` | 76 | Fourier coefficients of the character shape |
| `mfeat-fac` | 216 | Profile correlations |
| `mfeat-kar` | 64 | Karhunen-Loève coefficients |
| `mfeat-pix` | 240 | Pixel averages in 2×3 windows (rebuilds a 15×16 image) |
| `mfeat-zer` | 47 | Zernike moments |
| `mfeat-mor` | 6 | Morphological features |

The files have no label column: rows 0-199 are digit `0`, rows 200-399 are digit `1`, and so on.

## 2. Problem statement

1. **Classification:** train the five Part B algorithms (Random Forest, AdaBoost, Gradient Boosting, Bagging with a Decision Tree base, MLP) on the same data and split as the five Part A algorithms (Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVC). Compare all ten on Accuracy, weighted Precision/Recall/F1, confusion matrix and one-vs-rest ROC-AUC, then tune and select a final model.
2. **Clustering:** group the 649-dimensional digit features with **K-Means** (k chosen from the elbow curve) and **Agglomerative hierarchical clustering** (dendrogram, linkage comparison). Evaluate with Silhouette, Davies-Bouldin and Calinski-Harabasz, and visualise with PCA (2 components) and t-SNE. The ground-truth digits are used **only after fitting**, for validation.

## 3. Method

### Classification
- Stratified 80/20 split with `random_state=42`, the same split for all ten models.
- Every model is a `Pipeline(StandardScaler → model)`, so the scaler is fitted on the training data only, including inside each cross-validation fold.
- Metrics: Accuracy, weighted Precision/Recall/F1, one-vs-rest weighted ROC-AUC, and a confusion matrix for every model.
- Tuning: all ten models are tuned with 5-fold stratified `GridSearchCV` on weighted F1, using the parameters named in guideline 3.2 (e.g. `n_estimators`/`learning_rate` for AdaBoost, `hidden_layer_sizes`/`activation` for MLP, `C`/`kernel` for SVC). The final model is chosen by **cross-validated** F1, not by test score.

### Clustering
- All six feature sets are concatenated (649 features) and standardised.
- K-Means: elbow curve (inertia vs. k) for k = 2…15, with silhouette per k and a k-sensitivity table. The automatic knee estimate is k = 6, silhouette peaks at k = 9, and the k used is **10** (`K_FINAL` in `src/config.py`).
- Agglomerative: ward, complete, average and single linkage compared at the same k, with dendrograms for ward, complete and average.
- New samples (app only): K-Means uses `predict`. Agglomerative has no `predict`, so new points are assigned to the nearest cluster centroid.

## 4. Results

### Classification: consolidated table, all 10 algorithms after tuning (test set, n = 400)

| Model               | Part         |   Accuracy |   Precision (weighted) |   Recall (weighted) |   F1 (weighted) |   ROC-AUC (OvR) |
|:--------------------|:-------------|-----------:|-----------------------:|--------------------:|----------------:|----------------:|
| Logistic Regression | A (Review 1) |     0.9825 |                 0.9826 |              0.9825 |          0.9825 |          0.9991 |
| SVM (SVC)           | A (Review 1) |     0.9750 |                 0.9750 |              0.9750 |          0.9750 |          0.9994 |
| MLP (Neural Net)    | B (Review 2) |     0.9575 |                 0.9576 |              0.9575 |          0.9575 |          0.9980 |
| AdaBoost            | B (Review 2) |     0.9375 |                 0.9383 |              0.9375 |          0.9377 |          0.9933 |
| Gradient Boosting   | B (Review 2) |     0.9375 |                 0.9374 |              0.9375 |          0.9374 |          0.9923 |
| Bagging             | B (Review 2) |     0.8975 |                 0.8978 |              0.8975 |          0.8976 |          0.9877 |
| Random Forest       | B (Review 2) |     0.8875 |                 0.8876 |              0.8875 |          0.8872 |          0.9804 |
| Decision Tree       | A (Review 1) |     0.8375 |                 0.8367 |              0.8375 |          0.8370 |          0.9511 |
| Naive Bayes         | A (Review 1) |     0.8100 |                 0.8113 |              0.8100 |          0.8105 |          0.9506 |
| KNN                 | A (Review 1) |     0.7200 |                 0.7219 |              0.7200 |          0.7203 |          0.8792 |

**Final model: SVM (SVC)** (`C=10, kernel=linear`), selected on 5-fold cross-validated weighted F1.

| | Accuracy | F1 (weighted) | ROC-AUC (OvR) |
|:--|--:|--:|--:|
| Baseline | 0.8950 | 0.8956 | 0.9833 |
| **Tuned** | **0.9750** | **0.9750** | **0.9994** |

<details><summary>Baseline table (default settings, before tuning)</summary>

| Model               | Part         |   Accuracy |   Precision (weighted) |   Recall (weighted) |   F1 (weighted) |   ROC-AUC (OvR) |
|:--------------------|:-------------|-----------:|-----------------------:|--------------------:|----------------:|----------------:|
| Logistic Regression | A (Review 1) |     0.9650 |                 0.9650 |              0.9650 |          0.9650 |          0.9987 |
| MLP (Neural Net)    | B (Review 2) |     0.9275 |                 0.9286 |              0.9275 |          0.9275 |          0.9941 |
| Gradient Boosting   | B (Review 2) |     0.9125 |                 0.9124 |              0.9125 |          0.9123 |          0.9896 |
| Bagging             | B (Review 2) |     0.8975 |                 0.8978 |              0.8975 |          0.8976 |          0.9877 |
| SVM (SVC)           | A (Review 1) |     0.8950 |                 0.8969 |              0.8950 |          0.8956 |          0.9833 |
| Random Forest       | B (Review 2) |     0.8800 |                 0.8796 |              0.8800 |          0.8797 |          0.9767 |
| Decision Tree       | A (Review 1) |     0.8300 |                 0.8319 |              0.8300 |          0.8302 |          0.8867 |
| Naive Bayes         | A (Review 1) |     0.8100 |                 0.8113 |              0.8100 |          0.8105 |          0.9506 |
| KNN                 | A (Review 1) |     0.5000 |                 0.5211 |              0.5000 |          0.5054 |          0.7697 |
| AdaBoost            | B (Review 2) |     0.4900 |                 0.4831 |              0.4900 |          0.4437 |          0.8200 |

</details>

<details><summary>Tuning results (5-fold CV)</summary>

| Model               | Best params                                                              |   CV F1 (5-fold) |   CV F1 std |   F1 before |   F1 after |
|:--------------------|:-------------------------------------------------------------------------|-----------------:|------------:|------------:|-----------:|
| SVM (SVC)           | C=10, kernel=linear                                                      |           0.9650 |      0.0068 |      0.8956 |     0.9750 |
| Logistic Regression | C=100                                                                    |           0.9632 |      0.0054 |      0.9650 |     0.9825 |
| MLP (Neural Net)    | activation=tanh, hidden_layer_sizes=(128, 64)                            |           0.9549 |      0.0048 |      0.9275 |     0.9575 |
| AdaBoost            | estimator=DecisionTree(max_depth=3), learning_rate=0.5, n_estimators=200 |           0.9199 |      0.0180 |      0.4437 |     0.9377 |
| Gradient Boosting   | learning_rate=0.1, max_depth=2, n_estimators=300                         |           0.9034 |      0.0170 |      0.9123 |     0.9374 |
| Bagging             | max_features=1.0, max_samples=1.0, n_estimators=100                      |           0.8868 |      0.0185 |      0.8976 |     0.8976 |
| Random Forest       | max_depth=10, n_estimators=300                                           |           0.8728 |      0.0297 |      0.8797 |     0.8872 |
| Decision Tree       | max_depth=8, min_samples_leaf=10                                         |           0.8372 |      0.0158 |      0.8302 |     0.8370 |
| Naive Bayes         | var_smoothing=1e-11                                                      |           0.8044 |      0.0271 |      0.8105 |     0.8105 |
| KNN                 | metric=manhattan, n_neighbors=31                                         |           0.6639 |      0.0168 |      0.5054 |     0.7203 |

</details>

### Clustering (k = 10)

| Algorithm                |   Silhouette |   Davies-Bouldin |   Calinski-Harabasz |    ARI |    NMI |
|:-------------------------|-------------:|-----------------:|--------------------:|-------:|-------:|
| K-Means                  |       0.1325 |           2.2376 |            145.2921 | 0.8376 | 0.8569 |
| Agglomerative (ward)     |       0.1286 |           2.2943 |            139.3618 | 0.9026 | 0.9116 |
| Agglomerative (complete) |       0.0660 |           2.5599 |             86.9448 | 0.2962 | 0.4914 |
| Agglomerative (average)  |       0.0888 |           1.5535 |             61.0417 | 0.3095 | 0.5774 |
| Agglomerative (single)   |      -0.0160 |           0.8118 |              1.5586 | 0.0000 | 0.0089 |

ARI and NMI compare clusters with the true digits after fitting and are never used for training.

Linkage comparison (Agglomerative):

| Linkage   |   Silhouette |   Davies-Bouldin |   Calinski-Harabasz |   Largest cluster |   Smallest cluster |
|:----------|-------------:|-----------------:|--------------------:|------------------:|-------------------:|
| ward      |       0.1286 |           2.2943 |            139.3618 |               234 |                174 |
| complete  |       0.0660 |           2.5599 |             86.9448 |               483 |                 29 |
| average   |       0.0888 |           1.5535 |             61.0417 |               968 |                  1 |
| single    |      -0.0160 |           0.8118 |              1.5586 |              1991 |                  1 |

### Discussion

> _Team: write your interpretation of these results here in your own words (guideline 7.5). The notebooks contain the guiding questions._

Figures are in `reports/figures/`: elbow curve, dendrograms, PCA for every algorithm, t-SNE, metric bars, contingency heatmaps, classifier comparison, confusion matrices for all 10 models (baseline and tuned), and feature importance.

## 5. Repository structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv                 Mobile Price Classification
│   └── mfeat/                    UCI Multiple Features (6 files + mfeat.info)
├── notebooks/
│   ├── regression.ipynb          Review 1 (regression track)
│   ├── classification.ipynb      Part A + Part B, consolidated table, tuning
│   └── clustering.ipynb          K-Means, Agglomerative, metrics, PCA, t-SNE
├── src/                          Shared code used by notebooks, pipeline and app
│   ├── config.py                 Paths, seed, k, linkages
│   ├── data_loader.py            Loading and splitting both datasets
│   ├── classification.py         10 models, metrics, tuning grids
│   ├── clustering.py             Elbow, K-Means, Agglomerative, metrics, PCA/t-SNE
│   └── visualize.py              All figures
├── train_pipeline.py             End-to-end: train, evaluate, save models and reports
├── models/                       best_classifier.joblib, clustering.joblib
├── reports/                      CSV tables, results.json, figures/
└── app/
    ├── app.py                    Gradio app (bonus)
    └── ui/                       App components and stylesheet
```

## 6. Environment setup

Python 3.11 or newer is required (3.12 recommended).

```bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter                # only needed for the notebooks
```

## 7. How to run

```bash
# 1. Train everything (~5 min on a 2-core machine; writes models/ and reports/)
python train_pipeline.py
#    options: --only clf | --only clu | --no-tsne

# 2. Launch the web app -> http://127.0.0.1:7860
python app/app.py

# 3. Notebooks (run top to bottom)
jupyter notebook notebooks/
```

The trained models are already included, so step 1 is only needed after changing code or data.

### App
- **Home:** two cards, Mobile price predictor (left) and Handwritten digit clusters (right).
- **Mobile price predictor:** sliders for each spec, connectivity toggles, example presets and a random real phone. The prediction and class probabilities update live. The model comparison is in "How this model was chosen".
- **Handwritten digit clusters:** an interactive PCA/t-SNE map for any algorithm or linkage, the three metrics, an average-image gallery of each cluster, a digit inspector, cluster assignment for uploaded rows, and how k was chosen.

## 8. Deployment (Render, free tier)

The app runs as a free Render web service. It reads the `PORT` variable Render provides, and `.python-version` pins Python 3.12. It uses about 320 MB of memory, within the free tier's 512 MB.

1. Push this repository to GitHub.
2. On render.com, sign in with GitHub and choose **New → Web Service**, then pick this repository.
3. Settings:
   - **Language:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python app/app.py`
   - **Instance type:** Free
4. Click **Deploy**. The first build takes a few minutes, and the public URL appears at the top of the service page.

Free services sleep after 15 minutes without traffic and take about a minute to wake up, so open the link before a demo.

(Hugging Face Spaces now requires a PRO subscription to host Gradio apps, which is why Render is used. The YAML header at the top of this README is still valid if you have PRO.)

Public URL: _add after deploying_

## 9. AI assistance

As required by guideline 7.5: generative AI (Claude, by Anthropic) was used for **code scaffolding**, including the `src/` modules, `train_pipeline.py`, the Gradio app, notebook structure, and this README's structure and setup instructions. The analysis and interpretation (observation cells in the notebooks and the Discussion section above) are written by the team.

## 10. References

- Duin, R. P. W. *Multiple Features Data Set.* UCI Machine Learning Repository.
- Mobile Price Classification dataset (`train.csv`), as assigned by the instructor.
- scikit-learn, Gradio and Plotly documentation.
