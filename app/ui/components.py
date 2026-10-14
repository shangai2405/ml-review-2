"""HTML / Plotly building blocks for the Gradio UI. Pure functions, no Gradio state."""
import html

import numpy as np
import plotly.graph_objects as go

INK = "#1A1C3A"
MUTED = "#5B5F7E"
BLUE = "#3B5BFD"
MAGENTA = "#E03E7A"
BAND_COLORS = ["#16B39A", "#3B5BFD", "#7A3CF0", "#E03E7A"]   # low -> very high
# Matplotlib "tab10" - the colourblind-friendly palette recommended by guideline 7.3
CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# ------------------------------------------------------------------ pixel digits
def digit_svg(img, size=60, color=INK, cls="", style=""):
    """15x16 mfeat pixel image (values 0-6) -> crisp inline SVG."""
    img = np.asarray(img, float)
    vmax = max(img.max(), 1e-9)
    rects = []
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            v = img[r, c] / vmax
            if v > 0.04:
                rects.append(f'<rect x="{c}" y="{r}" width="1.02" height="1.02" '
                             f'fill-opacity="{v:.2f}"/>')
    h = size * img.shape[0] / img.shape[1]
    return (f'<svg class="{cls}" style="{style}" viewBox="0 0 {img.shape[1]} {img.shape[0]}" '
            f'width="{size}" height="{h:.0f}" fill="{color}" shape-rendering="crispEdges" '
            f'role="img" aria-label="handwritten digit">{"".join(rects)}</svg>')


# ------------------------------------------------------------------ landing page
def phone_card(best_name, acc):
    return f"""
<div class="land-card land-phone" tabindex="0" role="button" aria-label="Open the mobile price predictor">
  <div class="land-art">
    <svg viewBox="0 0 120 220" class="phone-svg" aria-hidden="true">
      <rect x="6" y="4" width="108" height="212" rx="20" class="phone-body"/>
      <rect x="14" y="22" width="92" height="170" rx="6" class="phone-screen"/>
      <rect x="46" y="10" width="28" height="5" rx="2.5" class="phone-notch"/>
      <rect x="24" y="36" width="54" height="7" rx="3.5" class="phone-line l1"/>
      <rect x="24" y="52" width="72" height="7" rx="3.5" class="phone-line l2"/>
      <rect x="24" y="68" width="40" height="7" rx="3.5" class="phone-line l3"/>
      <rect x="24" y="84" width="62" height="7" rx="3.5" class="phone-line l4"/>
      <rect x="24" y="120" width="72" height="56" rx="10" class="phone-tag"/>
      <text x="60" y="146" class="phone-tag-txt">₹ ? ? ?</text>
      <text x="60" y="164" class="phone-tag-sub">price band</text>
      <circle cx="60" cy="204" r="5" class="phone-btn"/>
    </svg>
  </div>
  <div class="land-text">
    <h2>Mobile price predictor</h2>
    <p>Set a phone's RAM, battery, screen and cameras, and see which of four price bands it falls into.</p>
    <div class="land-foot"><span class="pill pill-blue">Tuned {html.escape(best_name)}</span>
      <span class="pill">{acc*100:.1f}% test accuracy</span></div>
    <span class="land-cta">Open predictor</span>
  </div>
</div>"""


def digits_card(samples, ari):
    tiles = "".join(
        f'<div class="mosaic-cell" style="--i:{i}">{digit_svg(img, 34, MAGENTA if i % 7 == 3 else INK)}</div>'
        for i, img in enumerate(samples))
    return f"""
<div class="land-card land-digits" tabindex="0" role="button" aria-label="Open the digit clustering explorer">
  <div class="land-art"><div class="mosaic">{tiles}</div></div>
  <div class="land-text">
    <h2>Handwritten digit clusters</h2>
    <p>Watch K-Means and hierarchical clustering sort 2,000 handwritten digits into groups without ever seeing a label.</p>
    <div class="land-foot"><span class="pill pill-magenta">K-Means + Agglomerative</span>
      <span class="pill">ARI {ari:.2f} vs. true digits</span></div>
    <span class="land-cta">Open explorer</span>
  </div>
</div>"""


CARD_JS = """
const go = () => trigger('click');
element.addEventListener('click', go);
element.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
"""


# ------------------------------------------------------------------ price page
def price_result(proba, labels):
    k = int(np.argmax(proba))
    color = BAND_COLORS[k]
    rows = "".join(
        f"""<div class="prob-row{' is-top' if i == k else ''}">
              <span class="prob-name">{labels[i]}</span>
              <span class="prob-track"><span class="prob-fill" style="width:{p*100:.1f}%;background:{BAND_COLORS[i]}"></span></span>
              <span class="prob-val">{p*100:.0f}%</span></div>"""
        for i, p in enumerate(proba))
    steps = "".join(f'<span class="band-step{" on" if i <= k else ""}" style="--c:{BAND_COLORS[i]}"></span>'
                    for i in range(4))
    return f"""
<div class="result" style="--band:{color}">
  <div class="result-kicker">Predicted price band</div>
  <div class="result-band" key="{k}-{proba[k]:.3f}">{labels[k]}</div>
  <div class="band-steps" aria-hidden="true">{steps}</div>
  <div class="result-conf">{proba[k]*100:.0f}% confident</div>
  <div class="probs">{rows}</div>
</div>"""


def model_bars(comparison):
    names = [r["Model"] for r in comparison][::-1]
    f1 = [r["F1 (weighted)"] for r in comparison][::-1]
    part = [r["Part"] for r in comparison][::-1]
    colors = [BLUE if p.startswith("B") else "#AEB6D6" for p in part]
    fig = go.Figure(go.Bar(x=f1, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:.3f}" for v in f1], textposition="outside",
                           hovertemplate="%{y}<br>F1 %{x:.4f}<extra></extra>"))
    fig.update_layout(**_layout(), height=380, xaxis=dict(range=[0.6, 1.02], title="Weighted F1, test set (after tuning)",
                      gridcolor="#E6E9F4"), yaxis=dict(title=None),
                      title=dict(text="Blue: Part B (this review)   Grey: Part A (Review 1)",
                                 font=dict(size=12, color=MUTED), x=0.01))
    return fig


# ------------------------------------------------------------------ clustering page
def _layout():
    return dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Figtree, sans-serif", color=INK, size=13),
                margin=dict(l=10, r=10, t=40, b=10))


def cluster_scatter(coords, labels, y_true, method, title, highlight=None):
    fig = go.Figure()
    idx = np.arange(len(labels))
    for c in np.unique(labels):
        m = labels == c
        fig.add_trace(go.Scattergl(
            x=coords[m, 0], y=coords[m, 1], mode="markers", name=f"{c}",
            marker=dict(size=6, color=CLUSTER_COLORS[c % 10], opacity=.78,
                        line=dict(width=0)),
            customdata=np.c_[idx[m], y_true[m]],
            hovertemplate="Sample %{customdata[0]}<br>Group " + str(c) +
                          "<br>True digit %{customdata[1]}<extra></extra>"))
    if highlight is not None:
        fig.add_trace(go.Scatter(x=[coords[highlight, 0]], y=[coords[highlight, 1]], mode="markers",
                                 name="selected", marker=dict(size=22, color="rgba(0,0,0,0)",
                                 line=dict(width=3, color=INK)), hoverinfo="skip"))
    fig.update_layout(**_layout(), height=470, legend=dict(title="Group", itemsizing="constant"),
                      title=dict(text=title, font=dict(size=15), x=0.01),
                      xaxis=dict(title=f"{method} 1", gridcolor="#E6E9F4", zeroline=False),
                      yaxis=dict(title=f"{method} 2", gridcolor="#E6E9F4", zeroline=False))
    return fig


def metric_tiles(m):
    tiles = [
        ("Silhouette", f"{m['Silhouette']:.3f}", "Higher is better, range −1 to 1"),
        ("Davies-Bouldin", f"{m['Davies-Bouldin']:.3f}", "Lower is better"),
        ("Calinski-Harabasz", f"{m['Calinski-Harabasz']:.0f}", "Higher is better"),
        ("Agreement with true digits", f"{m['ARI']:.2f}", "ARI, checked after fitting"),
    ]
    return '<div class="tiles">' + "".join(
        f'<div class="tile{" tile-accent" if i == 3 else ""}"><div class="tile-label">{a}</div>'
        f'<div class="tile-value">{b}</div><div class="tile-hint">{c}</div></div>'
        for i, (a, b, c) in enumerate(tiles)) + "</div>"


def cluster_gallery(mean_imgs, mapping):
    cells = []
    for c, img in enumerate(mean_imgs):
        row = mapping.loc[c]
        pur = float(row["Purity"])
        cells.append(f"""
<div class="gal-cell" style="--c:{CLUSTER_COLORS[c % 10]}">
  <div class="gal-head"><span class="gal-dot"></span>Group {c}</div>
  {digit_svg(img, 64, CLUSTER_COLORS[c % 10])}
  <div class="gal-digit">mostly <b>{int(row['Majority digit'])}</b></div>
  <div class="gal-bar"><span style="width:{pur*100:.0f}%"></span></div>
  <div class="gal-meta">{pur*100:.0f}% pure, {int(row['Size'])} digits</div>
</div>""")
    return ('<div class="gal-intro">Each picture is the average of every digit in that group. '
            'Sharp shapes mean the group is consistent.</div><div class="gallery">' + "".join(cells) + "</div>")


def inspect_card(img, idx, km_c, km_d, ag_c, ag_d, linkage, truth):
    return f"""
<div class="inspect">
  <div class="inspect-img">{digit_svg(img, 150, INK)}</div>
  <div class="inspect-info">
    <div class="inspect-title">Sample {idx}</div>
    <div class="inspect-row"><span class="gal-dot" style="--c:{CLUSTER_COLORS[km_c]}"></span>
      K-Means put it in <b>group {km_c}</b>, which is mostly {km_d}s</div>
    <div class="inspect-row"><span class="gal-dot" style="--c:{CLUSTER_COLORS[ag_c]}"></span>
      Agglomerative ({linkage}) put it in <b>group {ag_c}</b>, which is mostly {ag_d}s</div>
    <div class="inspect-truth">It is actually a <b>{truth}</b>. The label was only used for this check, never for clustering.</div>
  </div>
</div>"""
