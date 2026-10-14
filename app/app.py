"""Gradio front-end for Review 2.

    python app/app.py        -> http://127.0.0.1:7860

Pages
  Home        two cards: Mobile price predictor (left) and Digit clusters (right)
  Price       live price-band prediction from phone specs
  Clusters    interactive PCA/t-SNE map, metrics, cluster gallery, digit inspector,
              new-sample assignment, and how k was chosen
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]      # app/ (for ui) and project root (for src)

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from src import clustering as clu
from src import config as C
from src import data_loader as dl
from ui import components as ui

# ------------------------------------------------------------------ load artefacts
if not (C.MODELS_DIR / "best_classifier.joblib").exists() or \
        not (C.MODELS_DIR / "clustering.joblib").exists():
    raise SystemExit("Models not found - run `python train_pipeline.py` first.")

CLF = joblib.load(C.MODELS_DIR / "best_classifier.joblib")
CLU = joblib.load(C.MODELS_DIR / "clustering.joblib")
RESULTS = json.loads((C.REPORTS_DIR / "results.json").read_text())
DF = dl.load_classification()
X_RAW, Y_TRUE, _ = dl.load_mfeat()
PIX = np.loadtxt(C.MFEAT_DIR / "mfeat-pix")
FEATURES = CLF["features"]
BAND_LABELS = [C.PRICE_LABELS[i] for i in range(4)]
CSS = (HERE / "ui" / "style.css").read_text()

# ------------------------------------------------------------------ price page logic
BINARY = {"blue": "Bluetooth", "dual_sim": "Dual SIM", "four_g": "4G", "three_g": "3G",
          "touch_screen": "Touch screen", "wifi": "Wi-Fi"}
GROUPS = {
    "Memory and speed": [("ram", "RAM (MB)", 1), ("int_memory", "Storage (GB)", 1),
                         ("clock_speed", "CPU clock (GHz)", .1), ("n_cores", "CPU cores", 1)],
    "Display": [("px_height", "Resolution height (px)", 1), ("px_width", "Resolution width (px)", 1),
                ("sc_h", "Screen height (cm)", 1), ("sc_w", "Screen width (cm)", 1)],
    "Battery and build": [("battery_power", "Battery (mAh)", 1), ("talk_time", "Talk time (hours)", 1),
                          ("mobile_wt", "Weight (g)", 1), ("m_dep", "Thickness (cm)", .1)],
    "Cameras": [("pc", "Main camera (MP)", 1), ("fc", "Front camera (MP)", 1)],
}
NUMERIC = [f for g in GROUPS.values() for f, _, _ in g]

PRESETS = {
    "Budget phone": dict(ram=700, int_memory=8, clock_speed=1.0, n_cores=4, px_height=400, px_width=800,
                         sc_h=10, sc_w=4, battery_power=900, talk_time=8, mobile_wt=170, m_dep=.8,
                         pc=5, fc=2, conn=["Dual SIM", "3G"]),
    "Mid-range": dict(ram=1800, int_memory=32, clock_speed=1.8, n_cores=6, px_height=800, px_width=1300,
                      sc_h=13, sc_w=6, battery_power=1300, talk_time=12, mobile_wt=150, m_dep=.5,
                      pc=12, fc=5, conn=["Bluetooth", "Dual SIM", "4G", "3G", "Touch screen", "Wi-Fi"]),
    "Upper mid-range": dict(ram=2400, int_memory=48, clock_speed=2.2, n_cores=8, px_height=900,
                            px_width=1400, sc_h=15, sc_w=7, battery_power=1400, talk_time=14,
                            mobile_wt=140, m_dep=.4, pc=16, fc=8,
                            conn=["Bluetooth", "Dual SIM", "4G", "3G", "Touch screen", "Wi-Fi"]),
    "Flagship": dict(ram=3800, int_memory=64, clock_speed=2.8, n_cores=8, px_height=1700, px_width=1900,
                     sc_h=17, sc_w=8, battery_power=1900, talk_time=18, mobile_wt=130, m_dep=.3,
                     pc=20, fc=16, conn=["Bluetooth", "Dual SIM", "4G", "3G", "Touch screen", "Wi-Fi"]),
}


def predict_price(*vals):
    nums, conn = vals[:-1], vals[-1] or []
    row = dict(zip(NUMERIC, map(float, nums)))
    row.update({f: float(lbl in conn) for f, lbl in BINARY.items()})
    proba = CLF["model"].predict_proba(pd.DataFrame([row])[FEATURES])[0]
    return ui.price_result(proba, BAND_LABELS)


def apply_preset(name):
    p = PRESETS[name]
    return [float(p[f]) for f in NUMERIC] + [p["conn"], f"Loaded the **{name.lower()}** example."]


def random_phone():
    r = DF.sample(1).iloc[0]
    conn = [lbl for f, lbl in BINARY.items() if r[f] == 1]
    note = f"Loaded a real phone from the dataset. Its actual price band is **{C.PRICE_LABELS[int(r[C.TARGET_COL])]}**."
    return [float(r[f]) for f in NUMERIC] + [conn, note]


# ------------------------------------------------------------------ clustering page logic
ALGOS = {"K-Means": None, "Ward": "ward", "Complete": "complete", "Average": "average", "Single": "single"}
ALGO_NOTES = {
    "K-Means": "Places 10 centres, assigns every digit to its nearest centre, and moves the centres until nothing changes.",
    "Ward": "Starts with every digit on its own and keeps merging the two groups whose merge adds the least variance.",
    "Complete": "Merges the two groups whose farthest members are closest. Sensitive to outliers.",
    "Average": "Merges the two groups with the smallest average distance between all their members.",
    "Single": "Merges the two groups whose closest members are nearest. It chains almost everything into one giant group.",
}


def labels_for(algo):
    lk = ALGOS[algo]
    return CLU["kmeans"].labels_ if lk is None else CLU["agg_labels"][lk]


def metrics_for(algo):
    name = "K-Means" if ALGOS[algo] is None else f"Agglomerative ({ALGOS[algo]})"
    return CLU["metrics"].set_index("Algorithm").loc[name]


MEAN_IMGS = {a: [PIX[labels_for(a) == c].mean(0).reshape(16, 15) for c in np.unique(labels_for(a))]
             for a in ALGOS}
MAPPINGS = {a: clu.map_clusters_to_digits(Y_TRUE, labels_for(a))[0] for a in ALGOS}


def cluster_page(algo, view, color, highlight=None):
    coords = CLU["tsne_xy"] if view == "t-SNE" and CLU["tsne_xy"] is not None else CLU["pca_xy"]
    method = "t-SNE" if coords is CLU["tsne_xy"] else "PCA"
    if color == "True digits":
        lab, title = Y_TRUE, "Coloured by the real digit (for comparison only)"
    else:
        lab, title = labels_for(algo), f"{algo}: 10 groups found without labels"
    fig = ui.cluster_scatter(coords, lab, Y_TRUE, method, title, highlight)
    side = ui.metric_tiles(metrics_for(algo)) + f'<p class="panel-sub" style="margin-top:14px">{ALGO_NOTES[algo]}</p>'
    return fig, side, ui.cluster_gallery(MEAN_IMGS[algo], MAPPINGS[algo])


def inspect(idx, algo, view, color):
    idx = int(idx)
    km_c = int(CLU["kmeans"].labels_[idx])
    lk = CLU["best_linkage"]
    ag_c = int(CLU["agg_labels"][lk][idx])
    card = ui.inspect_card(PIX[idx].reshape(16, 15), idx, km_c,
                           int(MAPPINGS["K-Means"].loc[km_c, "Majority digit"]), ag_c,
                           int(MAPPINGS["Ward"].loc[ag_c, "Majority digit"]), lk, int(Y_TRUE[idx]))
    fig, side, gal = cluster_page(algo, view, color, idx)
    return card, fig


def assign_new(file, n):
    if file is not None:
        path = str(file)
        X_new = np.loadtxt(path, ndmin=2, delimiter="," if path.endswith(".csv") else None)
        src = "your file"
    else:
        X_new = X_RAW[np.random.default_rng().choice(len(X_RAW), int(n), replace=False)]
        src = f"{int(n)} random digits from the dataset"
    if X_new.shape[1] != X_RAW.shape[1]:
        raise gr.Error(f"Each row needs {X_RAW.shape[1]} numbers (all six mfeat feature sets), "
                       f"but this file has {X_new.shape[1]}.")
    Xs = CLU["scaler"].transform(X_new)
    lk = CLU["best_linkage"]
    km = CLU["kmeans"].predict(Xs)
    ag = clu.nearest_centroid(Xs, CLU["agg_centroids"][lk])
    out = pd.DataFrame({"Row": np.arange(1, len(Xs) + 1), "K-Means group": km,
                        "Looks like (K-Means)": [int(MAPPINGS["K-Means"].loc[c, "Majority digit"]) for c in km],
                        f"Agglomerative ({lk}) group": ag,
                        "Looks like (Agglomerative)": [int(MAPPINGS["Ward"].loc[c, "Majority digit"]) for c in ag]})
    xy = CLU["pca"].transform(Xs)
    fig = ui.cluster_scatter(CLU["pca_xy"], CLU["kmeans"].labels_, Y_TRUE, "PCA",
                             "Where the new digits land (black rings)")
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", name="new",
                             text=[str(i) for i in range(1, len(xy) + 1)], textposition="top center",
                             marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=3, color=ui.INK))))
    return out, fig, f"Assigned {len(out)} rows from {src}."


def example_file():
    path = C.REPORTS_DIR / "sample_rows.csv"
    np.savetxt(path, X_RAW[np.random.default_rng(0).choice(2000, 5, replace=False)], delimiter=",", fmt="%.6g")
    return str(path)


def fig(name):
    p = C.FIG_DIR / name
    return str(p) if p.exists() else None


# ------------------------------------------------------------------ navigation
def show(page):
    return [gr.Column(visible=page == p) for p in ("home", "price", "clusters")]


# ------------------------------------------------------------------ layout
THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(c50="#EEF1FF", c100="#E8ECFF", c200="#C9D2FF", c300="#A3B2FF",
                                c400="#6F86FE", c500="#3B5BFD", c600="#2F4AE0", c700="#2538B3",
                                c800="#1C2A85", c900="#141E5E", c950="#0C1340"),
    neutral_hue="slate", radius_size="lg",
    font=[gr.themes.GoogleFont("Figtree"), "sans-serif"],
).set(body_background_fill="#F4F6FB", block_background_fill="#FFFFFF", block_border_width="1.5px",
      block_border_color="#E1E5F2", block_radius="18px", block_shadow="none",
      slider_color="#3B5BFD", button_primary_background_fill="#3B5BFD",
      button_primary_background_fill_hover="#2F4AE0", body_text_color="#1A1C3A")
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600..800'
        '&family=Figtree:wght@400;500;600;700&display=swap" rel="stylesheet">')
FORCE_LIGHT = "() => { document.body.classList.remove('dark'); }"

best_metrics = CLF["metrics"]
mosaic_idx = [d * 200 + o for o in (3, 41, 117) for d in range(10)][:18]
km_ari = float(metrics_for("K-Means")["ARI"])

with gr.Blocks(title="Phones and digits | ML Review 2") as demo:
    # ======================= HOME =======================
    with gr.Column(visible=True) as home:
        gr.HTML('<div class="hero"><h1>Phones and handwritten digits</h1>'
                '<p>Two machine-learning projects in one place. Predict what a phone costs from its specs, '
                'or explore how unlabelled handwritten digits sort themselves into groups.</p></div>')
        with gr.Row(equal_height=True, elem_classes="land-row"):
            card_price = gr.HTML(ui.phone_card(CLF["name"], best_metrics["Accuracy"]),
                                 js_on_load=ui.CARD_JS, padding=False, container=False)
            card_clu = gr.HTML(ui.digits_card([PIX[i].reshape(16, 15) for i in mosaic_idx], km_ari),
                               js_on_load=ui.CARD_JS, padding=False, container=False)

    # ======================= PRICE =======================
    with gr.Column(visible=False) as price_page:
        with gr.Row(elem_classes="page-head"):
            back1 = gr.Button("Home", elem_classes="back-btn", scale=0)
            gr.HTML('<div><h1>Mobile price predictor</h1><p>Move any slider and the prediction updates '
                    f'straight away. The model is a tuned {CLF["name"]} trained on 1,600 phones.</p></div>')
        with gr.Row(elem_classes="preset-row"):
            preset_btns = [gr.Button(n, elem_classes="chip", size="sm", scale=0, min_width=130) for n in PRESETS]
            rand_btn = gr.Button("Random real phone", elem_classes="chip", size="sm", scale=0, min_width=160)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                inputs = {}
                group_items = list(GROUPS.items())
                for row_groups in (group_items[:2], group_items[2:]):
                    with gr.Row(equal_height=True):
                        for gname, fields in row_groups:
                            with gr.Column(elem_classes="spec-group", min_width=240):
                                gr.HTML(f'<div class="panel-title">{gname}</div>')
                                for f, label, step in fields:
                                    inputs[f] = gr.Slider(float(DF[f].min()), float(DF[f].max()),
                                                          value=float(DF[f].median()), step=step, label=label)
                with gr.Column(elem_classes="box"):
                    gr.HTML('<div class="panel-title">Connectivity</div>')
                    conn = gr.CheckboxGroup(list(BINARY.values()), value=["Dual SIM", "4G", "3G", "Touch screen"],
                                            show_label=False, elem_classes="conn")
            with gr.Column(scale=2, elem_classes="sticky-col"):
                result = gr.HTML(container=False, padding=False)
                note = gr.Markdown(elem_classes="truth-note")
        with gr.Accordion("How this model was chosen", open=False):
            gr.Markdown("All 10 algorithms were trained on the same 80/20 split, then each was tuned with "
                        "5-fold cross-validated GridSearchCV. The winner was picked on cross-validated "
                        f"weighted F1: **{CLF['name']}** ({RESULTS['classification']['best_params']}).")
            gr.Plot(ui.model_bars(RESULTS["classification"]["final"]), show_label=False)
            gr.Dataframe(pd.DataFrame(RESULTS["classification"]["final"]), interactive=False,
                         label="All 10 models after tuning (test set)")
            gr.Dataframe(pd.DataFrame(RESULTS["classification"]["tuning"]), interactive=False,
                         label="Tuning: before and after")

    # ======================= CLUSTERS =======================
    with gr.Column(visible=False) as clu_page:
        with gr.Row(elem_classes="page-head"):
            back2 = gr.Button("Home", elem_classes="back-btn", scale=0)
            gr.HTML('<div><h1>Handwritten digit clusters</h1><p>2,000 digits described by 649 numbers each. '
                    'The algorithms never see which digit is which. Hover over any point to see what it really is.</p></div>')
        with gr.Row():
            algo = gr.Radio(list(ALGOS), value="K-Means", label="Algorithm", elem_classes="seg", scale=3)
            view = gr.Radio(["PCA", "t-SNE"], value="t-SNE", label="Map", elem_classes="seg", scale=1)
            color = gr.Radio(["Groups found", "True digits"], value="Groups found", label="Colour by",
                             elem_classes="seg", scale=2)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes="box"):
                scatter = gr.Plot(show_label=False)
            with gr.Column(scale=2):
                side = gr.HTML(container=False, padding=False)
        with gr.Column(elem_classes="box"):
            gr.HTML('<div class="panel-title">What each group contains</div>')
            gallery = gr.HTML(container=False, padding=False)

        with gr.Tabs():
            with gr.Tab("Inspect a digit"):
                with gr.Row():
                    idx = gr.Slider(0, 1999, value=1250, step=1, label="Pick any of the 2,000 digits", scale=4)
                    rnd_idx = gr.Button("Surprise me", elem_classes="chip", scale=1)
                inspect_out = gr.HTML(container=False, padding=False)
            with gr.Tab("Group new digits"):
                gr.Markdown("Upload a CSV with **649 numbers per row** (the six mfeat feature sets in order: "
                            "fou, fac, kar, pix, zer, mor), or try random digits from the dataset. K-Means uses "
                            "its own `predict`; Agglomerative has none, so new digits join the nearest group centre.")
                with gr.Row():
                    up = gr.File(label="Your rows", file_types=[".csv", ".txt"])
                    with gr.Column():
                        n = gr.Slider(1, 20, value=5, step=1, label="Or this many random digits")
                        ex_btn = gr.Button("Download an example file", elem_classes="chip")
                        ex_file = gr.File(label="Example file", visible=True)
                assign_btn = gr.Button("Group these digits", variant="primary")
                assign_msg = gr.Markdown()
                with gr.Row():
                    assign_tab = gr.Dataframe(interactive=False)
                    assign_plot = gr.Plot(show_label=False)
            with gr.Tab("How k was chosen"):
                cl = RESULTS["clustering"]
                gr.Markdown(f"The elbow curve bends gradually. An automatic knee detector suggests "
                            f"k = {cl['k_auto_knee']} and silhouette peaks at k = {cl['k_best_silhouette']}. "
                            f"The gain from each extra group drops off around k = 10, and the dataset is "
                            f"documented as 10 digit classes, so **k = {cl['k_used']}** is used.")
                with gr.Row():
                    gr.Image(fig("clu_elbow.png"), show_label=False, buttons=["fullscreen"])
                    gr.Image(fig("clu_dendrogram_ward.png"), show_label=False, buttons=["fullscreen"])
                gr.Dataframe(pd.DataFrame(cl["k_sensitivity"]), interactive=False, label="Metrics for each candidate k")
                gr.Dataframe(pd.DataFrame(cl["metrics"]), interactive=False, label="All algorithms at k = 10")
                with gr.Row():
                    gr.Image(fig("clu_dendrogram_complete.png"), show_label=False, buttons=["fullscreen"])
                    gr.Image(fig("clu_dendrogram_average.png"), show_label=False, buttons=["fullscreen"])

    # ======================= events =======================
    pages = [home, price_page, clu_page]
    price_inputs = [inputs[f] for f in NUMERIC] + [conn]

    card_price.click(lambda: show("price"), None, pages).then(predict_price, price_inputs, result)
    card_clu.click(lambda: show("clusters"), None, pages).then(
        cluster_page, [algo, view, color], [scatter, side, gallery]).then(
        inspect, [idx, algo, view, color], [inspect_out, scatter])
    for b in (back1, back2):
        b.click(lambda: show("home"), None, pages)

    for comp in price_inputs:
        comp.change(predict_price, price_inputs, result, show_progress="hidden")
    for b in preset_btns:
        b.click(apply_preset, gr.State(b.value), price_inputs + [note])
    rand_btn.click(random_phone, None, price_inputs + [note])

    for comp in (algo, view, color):
        comp.change(cluster_page, [algo, view, color], [scatter, side, gallery])
    idx.release(inspect, [idx, algo, view, color], [inspect_out, scatter])
    rnd_idx.click(lambda: int(np.random.randint(0, 2000)), None, idx).then(
        inspect, [idx, algo, view, color], [inspect_out, scatter])
    assign_btn.click(assign_new, [up, n], [assign_tab, assign_plot, assign_msg])
    ex_btn.click(example_file, None, ex_file)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), theme=THEME, css=CSS, head=HEAD, js=FORCE_LIGHT)
