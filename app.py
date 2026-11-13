"""Yield monitoring dashboard.

    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import ListedColormap

from yielddash import queries, spc, yieldmodel
from yielddash.etl import decode_map

ROOT = Path(__file__).parent
DATABASES = {"WM-811K (real fab data)": ROOT / "data" / "yield.db",
             "Synthetic demo": ROOT / "data" / "demo.db"}
MIN_LOTS = 50  # products with fewer lots are too short a history to chart
STEPS = {"first": "First test (use this for monitoring)",
         "repeat": "Repeat record of the same wafers",
         "retest": "Retest record (few fails)"}

# Ink and surfaces
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Series colours, always assigned in this order; red is reserved for out-of-control signals
BLUE, ORANGE, AQUA, YELLOW, CRITICAL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d03b3b"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def styled(fig: go.Figure, height: int, x_title: str, y_title: str, percent_y: bool = True) -> go.Figure:
    axis = dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", tickfont=dict(color=MUTED),
                title_font=dict(color=INK_2, size=13))
    fig.update_layout(
        template="none", height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13), margin=dict(l=64, r=48, t=40, b=52),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FONT, color=INK)),
        xaxis=dict(title=x_title, **axis), yaxis=dict(title=y_title, **axis))
    if percent_y:
        fig.update_yaxes(tickformat=".0%")
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False})


@st.cache_resource
def connection(path: str):
    return queries.connect(path)


@st.cache_data
def load_products(path: str) -> pd.DataFrame:
    return queries.products(connection(path), MIN_LOTS)


@st.cache_data
def load_trend(path: str, product: str, min_wafers: int, step: str) -> pd.DataFrame:
    return queries.lot_trend(connection(path), product, min_wafers, step)


@st.cache_data
def load_pareto(path: str, product: str, step: str) -> tuple[pd.DataFrame, pd.Series]:
    con = connection(path)
    return queries.pattern_pareto(con, product, step), queries.label_coverage(con, product, step)


# ------------------------------------------------------------------ charts

def spc_figure(trend: pd.DataFrame, chart: pd.DataFrame, baseline: int | None) -> go.Figure:
    center, sigma = chart.attrs["center"], chart.attrs["sigma"]
    hover = "Lot %{customdata[0]}<br>Yield %{y:.1%}<br>%{customdata[1]} wafers<extra></extra>"
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=trend["run"], y=trend["yield"], mode="lines+markers", name="Lot yield",
        line=dict(color=BLUE, width=1.5), marker=dict(color=BLUE, size=4),
        customdata=trend[["lot_id", "n_wafers"]], hovertemplate=hover))
    rule_text = chart["signal"].map(lambda r: f"Rule {r}: {spc.RULES[r]}" if r else "")
    signal = chart["signal"].to_numpy() > 0
    # A low-yield signal is the alarm. A high-side signal is worth knowing about but is not one,
    # so it gets a quiet hollow marker instead of the alarm colour.
    for side, name, marker in (
            ("low", "Low-yield signal", dict(color=CRITICAL, size=9, line=dict(color=SURFACE, width=1.5))),
            ("high", "High-side signal", dict(color=SURFACE, size=7, line=dict(color=INK_2, width=1.5)))):
        pick = signal & (chart["side"].to_numpy() == side)
        fig.add_trace(go.Scatter(
            x=trend.loc[pick, "run"], y=trend.loc[pick, "yield"], mode="markers", name=name, marker=marker,
            customdata=np.column_stack([trend.loc[pick, "lot_id"], rule_text[pick]]),
            hovertemplate="Lot %{customdata[0]}<br>Yield %{y:.1%}<br>%{customdata[1]}<extra></extra>"))
    for level, label in ((center + 3 * sigma, "UCL"), (center, "CL"), (center - 3 * sigma, "LCL")):
        if 0 <= level <= 1:
            fig.add_hline(y=level, line=dict(color=MUTED, width=1), annotation_text=f"{label} {level:.1%}",
                          annotation_position="right", annotation_font=dict(color=INK_2, size=12))
    if baseline and baseline < len(trend):
        fig.add_vline(x=baseline + 0.5, line=dict(color=AXIS, width=1), annotation_text="end of baseline",
                      annotation_position="top left", annotation_font=dict(color=INK_2, size=12))
    styled(fig, 420, "Lot (in lot-number order)", "Lot yield")
    fig.update_layout(margin=dict(r=96))
    fig.update_yaxes(range=[max(0, min(trend["yield"].min(), center - 3.5 * sigma) - 0.01), 1.0])
    return fig


def zone_figure(trend: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, name, color in (("edge_fail_rate", "Edge dies (outer 20% of the radius)", BLUE),
                                ("inner_fail_rate", "Inner dies", ORANGE)):
        smooth = trend[column].rolling(10, min_periods=1).mean()
        fig.add_trace(go.Scattergl(x=trend["run"], y=smooth, mode="lines", name=name,
                                   line=dict(color=color, width=2),
                                   hovertemplate="Lot run %{x}<br>" + name + " %{y:.1%}<extra></extra>"))
    return styled(fig, 300, "Lot (in lot-number order)", "Die fail rate, 10-lot average")


def pareto_figure(pareto: pd.DataFrame) -> go.Figure:
    d = pareto.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["fail_dies"], y=d["pattern"], orientation="h", marker=dict(color=BLUE, cornerradius=4),
        text=[f"{s:.0%}" for s in d["share"]], textposition="outside", textfont=dict(color=INK_2),
        cliponaxis=False, customdata=d[["n_wafers", "cum_share"]],
        hovertemplate="%{y}<br>%{x:,} failing dies on %{customdata[0]:,} wafers"
                      "<br>Cumulative share %{customdata[1]:.0%}<extra></extra>"))
    styled(fig, 80 + 38 * len(d), "Failing dies on wafers with this pattern", "", percent_y=False)
    fig.update_layout(bargap=0.35, margin=dict(l=96, r=64))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    return fig


def model_figure(points: pd.DataFrame, fits: pd.DataFrame, selected: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=points["area"], y=points["yield"], mode="markers", name=f"Products with {MIN_LOTS}+ lots",
        marker=dict(color=MUTED, size=8, opacity=0.7, line=dict(color=SURFACE, width=1)),
        customdata=points[["product_id", "n_wafers"]],
        hovertemplate="%{customdata[0]}<br>Die area %{x:.2f} cm²<br>Yield %{y:.1%}"
                      "<br>%{customdata[1]:,} wafers<extra></extra>"))
    grid = np.linspace(points["area"].min() * 0.9, points["area"].max() * 1.05, 200)
    for (_, row), color in zip(fits.iterrows(), (BLUE, ORANGE, AQUA, YELLOW)):
        f = yieldmodel.MODELS[row["model"]]
        params = (row["d0"], row["alpha"]) if row["model"] == "Negative binomial" else (row["d0"],)
        fig.add_trace(go.Scatter(x=grid, y=f(grid, *params), mode="lines", name=row["model"],
                                 line=dict(color=color, width=2),
                                 hovertemplate=row["model"] + "<br>%{x:.2f} cm²: %{y:.1%}<extra></extra>"))
    here = points[points["product_id"] == selected]
    fig.add_trace(go.Scatter(
        x=here["area"], y=here["yield"], mode="markers+text", name="Selected product", text=[selected],
        textposition="top center", textfont=dict(color=INK), hoverinfo="skip",
        marker=dict(color=INK, size=11, line=dict(color=SURFACE, width=2))))
    return styled(fig, 440, "Approximate die area (cm²)", "Product die yield")


def wafer_gallery(wafers: pd.DataFrame):
    cmap = ListedColormap([SURFACE, GRID, CRITICAL])  # no die / pass / fail
    columns = 5
    rows = int(np.ceil(len(wafers) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.1 * columns, 2.35 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, w) in zip(axes.ravel(), wafers.iterrows()):
        ax.imshow(decode_map(w["map"], w["n_rows"], w["n_cols"]), cmap=cmap, vmin=0, vmax=2,
                  interpolation="nearest")
        label = f"Slot {w['slot']} · {w['yield']:.0%}"
        if w["pattern"] and w["pattern"] != "none":
            label += f"\n{w['pattern']}"
        ax.set_title(label, fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------- page

st.set_page_config(page_title="Yield monitor", layout="wide")

available = {name: path for name, path in DATABASES.items() if path.exists()}
if not available:
    st.error("No database found. Build one first: `python -m yielddash.etl --synthetic --db data/demo.db`")
    st.stop()

with st.sidebar:
    st.header("Filters")
    db_path = str(available[st.selectbox("Data source", list(available))])
    catalog = load_products(db_path)
    product = st.selectbox(
        "Product", catalog["product_id"],
        format_func=lambda p: f"{p} · {catalog.set_index('product_id').loc[p, 'n_lots']:,} lots")
    step = st.selectbox(
        "Test record", list(STEPS), format_func=STEPS.get,
        help="Consecutive lot numbers often hold the same wafers recorded again at a later test step. "
             "This is inferred from the wafer maps, not documented in the data. "
             "Monitoring uses the first record of each lot.")
    min_wafers = st.number_input("Minimum wafers per lot", 1, 25, 5,
                                 help="Lots with fewer wafers give a noisy yield and are left off the chart.")
    trend = load_trend(db_path, product, min_wafers, step)
    detached = spc.detached_top_cluster(trend["yield"]) if step == "first" and len(trend) else np.zeros(0, bool)
    n_hidden = 0
    if detached.any():
        hide = st.checkbox(
            f"Hide {int(detached.sum()):,} near-zero-fail lots", True,
            help="Lots at or above 99.5% yield that sit in a cluster of their own, with an empty band "
                 "between them and every normal lot. They are most likely later test records that "
                 "could not be linked to their first test.")
        if hide:
            n_hidden = int(detached.sum())
            trend = trend[~detached].reset_index(drop=True)
            trend["run"] = np.arange(1, len(trend) + 1)
    if len(trend) < 20:
        st.warning("Fewer than 20 lots match these filters. Change them or pick another product.")
        st.stop()
    baseline = st.number_input(
        "Baseline lots for control limits", 0, len(trend), min(100, len(trend) // 2), step=10,
        help="Limits are set from the first N lots and then held fixed. 0 uses every lot.")
    diameter = st.selectbox("Assumed wafer diameter (mm)", [300, 200, 150], index=0,
                            help="WM-811K does not state it. It only rescales die area and D0.")

chart = spc.control_chart(trend["yield"], baseline or None)
low = (chart["signal"] > 0) & (chart["side"] == "low")
high = (chart["signal"] > 0) & (chart["side"] == "high")

st.title("Yield monitor")
st.caption(f"Product {product} · {STEPS[step].split(' (')[0].lower()} · lots in lot-number order"
           + (f" · {n_hidden:,} near-zero-fail lots hidden" if n_hidden else ""))

k1, k2, k3, k4 = st.columns(4)
k1.metric("Lots charted", f"{len(trend):,}")
k2.metric("Wafers", f"{int(trend['n_wafers'].sum()):,}")
k3.metric("Die yield", f"{1 - trend['n_fail'].sum() / trend['n_die'].sum():.1%}",
          help="Passing dies / all dies over the charted lots.")
k4.metric("Lots with a low-yield signal", f"{int(low.sum()):,} ({low.mean():.1%})")

tab_spc, tab_pareto, tab_lot, tab_model = st.tabs(["Control chart", "Defect Pareto", "Lot drill-down", "Yield model"])

with tab_spc:
    st.subheader("Lot yield, individuals chart")
    st.caption("Limits are the baseline mean ± 3 sigma, with sigma estimated from the moving range. "
               "Marked points complete one of the four Western Electric run rules: red below the centre "
               "line, hollow above it.")
    show(spc_figure(trend, chart, baseline or None))

    counts = pd.DataFrame({
        "Rule": [f"{r}. {text.split(',')[0]}" for r, text in spc.RULES.items()],
        "Low side": [int((chart[f"rule{r}"] & (chart["side"] == "low")).sum()) for r in spc.RULES],
        "High side": [int((chart[f"rule{r}"] & (chart["side"] == "high")).sum()) for r in spc.RULES],
    })
    left, right = st.columns(2)
    with left:
        st.markdown("**Signals by rule**")
        st.dataframe(counts, hide_index=True, width="stretch")
        st.caption(f"{int(high.sum()):,} lots signal on the high side. A lot can trip more than one rule.")
    with right:
        st.markdown("**Lots with a low-yield signal, worst first**")
        flagged = trend.loc[low, ["lot_id", "run", "n_wafers", "yield", "min_wafer_yield"]].assign(
            rule=chart.loc[low, "signal"].map(spc.RULES)).sort_values("yield")
        st.dataframe(flagged, hide_index=True, width="stretch", height=212, column_config={
            "lot_id": "Lot", "run": "Run", "n_wafers": "Wafers", "rule": "First rule tripped",
            "yield": st.column_config.NumberColumn("Lot yield", format="percent"),
            "min_wafer_yield": st.column_config.NumberColumn("Worst wafer", format="percent")})

    st.subheader("Where on the wafer the fails are")
    st.caption("If the two lines separate, the loss is concentrated at the wafer edge or in the interior, "
               "which narrows down the process step to look at.")
    show(zone_figure(trend))

with tab_pareto:
    pareto, coverage = load_pareto(db_path, product, step)
    st.subheader("Yield loss by defect pattern")
    st.caption(f"{int(coverage['labeled']):,} of {int(coverage['wafers']):,} wafer records of this kind carry an "
               f"engineer's label, and {int(coverage['patterned']):,} of those show a pattern. "
               "The chart covers only those patterned wafers.")
    if pareto.empty:
        st.info("No labeled defect patterns for this product.")
    else:
        show(pareto_figure(pareto))
        st.dataframe(pareto, hide_index=True, width="stretch", column_config={
            "pattern": "Pattern", "n_wafers": "Wafers", "fail_dies": "Failing dies",
            "share": st.column_config.NumberColumn("Share of loss", format="percent"),
            "cum_share": st.column_config.NumberColumn("Cumulative", format="percent")})

with tab_lot:
    st.subheader("Wafer maps of one lot")
    order = trend.assign(signal=chart["signal"].to_numpy(), side=chart["side"].to_numpy()).sort_values("yield")
    describe = {r.lot_id: f"{r.lot_id} · {r['yield']:.1%}" + (f" · rule {r.signal} ({r.side})" if r.signal else "")
                for _, r in order.iterrows()}
    lot_id = st.selectbox("Lot (lowest yield first)", list(describe), format_func=describe.get)
    wafers = queries.lot_wafers(connection(db_path), lot_id)
    wafers = wafers[wafers["product_id"] == product]
    st.caption(f"{len(wafers)} wafers · lot yield {describe[lot_id].split(' · ')[1]} · red dies failed")
    st.pyplot(wafer_gallery(wafers), width="content")
    st.dataframe(wafers[["slot", "n_die", "n_fail", "yield", "pattern"]], hide_index=True, column_config={
        "slot": "Slot", "n_die": "Dies", "n_fail": "Failing dies", "pattern": "Label",
        "yield": st.column_config.NumberColumn("Yield", format="percent")})

with tab_model:
    st.subheader("Yield against die area, all products")
    points = catalog.assign(area=yieldmodel.die_area_cm2(catalog["die_count"], diameter))
    if len(points) < 3:
        st.info("At least three products are needed to fit yield against die area.")
    else:
        fits = yieldmodel.fit_models(points["area"], points["yield"], points["n_wafers"])
        st.caption("Each grey point is one product. Larger dies catch more defects, so yield falls with die "
                   "area; the four curves are the classic models fitted to these points, weighted by wafer count.")
        show(model_figure(points, fits, product))
        table = fits.assign(alpha=fits["alpha"].map(lambda a: "–" if pd.isna(a) else f"{a:.2f}"))
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "model": "Model",
            "d0": st.column_config.NumberColumn("D0 (defects per cm²)", format="%.3f"),
            "alpha": "Cluster factor alpha",
            "rmse": st.column_config.NumberColumn("Fit error (yield points)", format="%.3f")})
        here = points.set_index("product_id").loc[product]
        alpha = fits.set_index("model").loc["Negative binomial", "alpha"]
        d1, d2 = st.columns(2)
        d1.metric(f"{product}: implied D0, Poisson",
                  f"{yieldmodel.implied_d0(here['yield'], here['area']):.3f} /cm²")
        d2.metric(f"{product}: implied D0, negative binomial (alpha {alpha:.2f})",
                  f"{yieldmodel.implied_d0(here['yield'], here['area'], alpha):.3f} /cm²")
        st.caption(f"Assumes a {diameter} mm wafer with 3 mm edge exclusion and die area = usable area / die "
                   "count. Products differ in process and maturity, so one curve across all of them is a "
                   "rough description, not a process model.")
