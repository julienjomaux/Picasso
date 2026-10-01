"""
PICASSO aFRR CBMP visualizer — V5
Gem Energy Analytics · Julien Jomaux

Data: TransnetBW PICASSO CBMP API (4-second cross-border marginal prices).
"""
import io
import time
from datetime import datetime, date, time as dtime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import urllib3
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="Picasso aFRR prices - Visualizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =================================================================
# CONSTANTS
# =================================================================
LOCAL_TZ = ZoneInfo("Europe/Brussels")
API_URL = "https://api.transnetbw.de/picasso-cbmp/csv?date={d}&lang=de"
BRAND = "#0b6a6a"

POSTS = {
    "picasso": ("PICASSO: insights and data",
                "https://gemenergyanalytics.substack.com/p/picasso-insights-and-data"),
    "overlap": ("Overlapping aFRR merit orders and their impact on CBMP",
                "https://gemenergyanalytics.substack.com/p/overlapping-afrr-merit-orders-and"),
    "overlap2": ("Going further on overlapping aFRR merit orders",
                 "https://gemenergyanalytics.substack.com/p/going-further-on-overlapping-afrr"),
    "intro": ("An intro to the European balancing world",
              "https://gemenergyanalytics.substack.com/p/an-intro-to-the-european-balancing"),
    "reserves": ("European power reserves: part 2 – aFRR",
                 "https://gemenergyanalytics.substack.com/p/european-power-reserves-part-2-afrr"),
    "imbalance": ("The impact of aFRR on imbalance prices",
                  "https://gemenergyanalytics.substack.com/p/the-impact-of-afrr-on-imbalance-prices"),
}


def post_link(key):
    title, url = POSTS[key]
    return f"[{title}]({url})"


# code -> (country code, TSO name, country). Unknown codes found in the API
# are still shown automatically, with their raw code as label.
TSO_INFO = {
    "ELIA":    ("BE", "Elia", "Belgium"),
    "RTE":     ("FR", "RTE", "France"),
    "50HZT":   ("DE", "50Hertz", "Germany"),
    "AMP":     ("DE", "Amprion", "Germany"),
    "TNG":     ("DE", "TransnetBW", "Germany"),
    "TTG":     ("DE", "TenneT DE", "Germany"),
    "TNL":     ("NL", "TenneT NL", "Netherlands"),
    "APG":     ("AT", "APG", "Austria"),
    "SG":      ("CH", "Swissgrid", "Switzerland"),
    "CEPS":    ("CZ", "ČEPS", "Czechia"),
    "SEPS":    ("SK", "SEPS", "Slovakia"),
    "PSE":     ("PL", "PSE", "Poland"),
    "MAVIR":   ("HU", "MAVIR", "Hungary"),
    "ELES":    ("SI", "ELES", "Slovenia"),
    "TERNA":   ("IT", "Terna", "Italy"),
    "REE":     ("ES", "Red Eléctrica", "Spain"),
    "ADMIE":   ("GR", "IPTO", "Greece"),
    "ESO":     ("BG", "ESO", "Bulgaria"),
    "ENDK1":   ("DK1", "Energinet", "Denmark West"),
    "ENDK2":   ("DK2", "Energinet", "Denmark East"),
    "FINGRID": ("FI", "Fingrid", "Finland"),
    "ELERING": ("EE", "Elering", "Estonia"),
    "AST":     ("LV", "AST", "Latvia"),
    "LITGRID": ("LT", "Litgrid", "Lithuania"),
}
DEFAULT_TSOS = ["ELIA", "RTE", "50HZT"]
GERMAN_TSOS = ["50HZT", "AMP", "TNG", "TTG"]

PALETTE = [
    "#0b6a6a", "#3b6fd1", "#e08214", "#8e44ad", "#c0392b", "#16a085",
    "#d4ac0d", "#7f8c8d", "#e84393", "#2c3e50", "#27ae60", "#a0522d",
    "#5dade2", "#f39c12", "#6c3483", "#1abc9c", "#b03a2e", "#566573",
    "#ff7f50", "#2e86c1", "#76448a", "#229954", "#ca6f1e", "#34495e",
]
FIXED_COLORS = {"ELIA": "#0b6a6a", "RTE": "#3b6fd1", "50HZT": "#e08214"}

CAT_UP, CAT_DOWN, CAT_NEUTRAL = 1, 2, 3
DIR = {
    CAT_UP:      dict(key="up", label="Up", color="#d6443a", desc="Up only (aFRR up)"),
    CAT_DOWN:    dict(key="down", label="Down", color="#2e9e5b", desc="Down only (aFRR down)"),
    CAT_NEUTRAL: dict(key="neutral", label="No activation", color="#3b6fd1", desc="No activation (up & down prices both published)"),
}

PLOT_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def tso_label(code):
    cc, name, _ = TSO_INFO.get(code, ("??", code, "unknown"))
    return f"{cc} · {name}" if code in TSO_INFO else f"{code} (new/unknown)"


def tso_sort_key(code):
    order = list(TSO_INFO)
    return (order.index(code) if code in order else 999, code)


# =================================================================
# STYLE
# =================================================================
st.markdown(
    f"""
<style>
    .block-container {{padding-top: 2rem; padding-bottom: 3rem;}}
    .hero {{
        background: linear-gradient(120deg, {BRAND} 0%, #13908f 100%);
        color: #fff; padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;
    }}
    .hero h1 {{color:#fff; font-size: 1.9rem; margin: 0 0 .3rem 0; padding:0;}}
    .hero p {{color:#e6f4f4; margin: 0; font-size: 1rem;}}
    .hero a {{color:#fff; font-weight:600;}}
    .pill {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem;
            font-weight:600; color:#fff; margin-right:6px;}}
    h2 {{border-bottom: 2px solid {BRAND}22; padding-bottom: .3rem;}}
    div[data-testid="stMetricValue"] {{font-size: 1.35rem;}}
</style>
""",
    unsafe_allow_html=True,
)

# =================================================================
# DATA LOADING (vectorised + cached)
# =================================================================
_SESSION = requests.Session()


@st.cache_data(show_spinner=False, max_entries=40)
def load_day(date_str: str, cache_bucket: int):
    """Download one day of CBMP and pre-compute price + direction per TSO.

    `cache_bucket` only serves to refresh today's data every 5 minutes;
    past days are cached for the lifetime of the app.
    """
    url = API_URL.format(d=date_str)
    try:
        r = _SESSION.get(url, timeout=90)
    except requests.exceptions.SSLError:
        r = _SESSION.get(url, timeout=90, verify=False)
    if r.status_code != 200:
        return {"error": f"API error {r.status_code}"}
    if not r.content.strip():
        return {"error": "Empty response"}

    df = pd.read_csv(io.BytesIO(r.content), sep=";", na_values=["N/A"], low_memory=False)
    tcol = df.columns[0]
    t_local = (
        pd.to_datetime(df[tcol], utc=True, format="ISO8601")
        .dt.tz_convert(LOCAL_TZ)
        .dt.tz_localize(None)
        .to_numpy()
    )
    n = len(df)

    codes = []
    for c in df.columns[1:]:
        if c.endswith("_POS") or c.endswith("_NEG"):
            code = c[:-4]
            if code not in codes:
                codes.append(code)

    nan_col = np.full(n, np.nan)
    values, cats, pos_d, neg_d, empty = {}, {}, {}, {}, []
    for code in codes:
        pos = pd.to_numeric(df[f"{code}_POS"], errors="coerce").to_numpy(float) if f"{code}_POS" in df else nan_col
        neg = pd.to_numeric(df[f"{code}_NEG"], errors="coerce").to_numpy(float) if f"{code}_NEG" in df else nan_col
        hp, hn = ~np.isnan(pos), ~np.isnan(neg)
        if not (hp | hn).any():
            empty.append(code)
            continue
        cat = np.zeros(n, dtype=np.int8)
        cat[hp & ~hn] = CAT_UP
        cat[hn & ~hp] = CAT_DOWN
        cat[hp & hn] = CAT_NEUTRAL
        val = np.where(hp & hn, (pos + neg) / 2.0, np.where(hp, pos, neg))
        values[code], cats[code] = val, cat
        pos_d[code], neg_d[code] = pos, neg

    return {"times": t_local, "values": values, "cats": cats,
            "pos": pos_d, "neg": neg_d, "empty": empty, "n": n}


# =================================================================
# HELPERS
# =================================================================
def seconds_of_day(times):
    return ((times - times.astype("datetime64[D]")) / np.timedelta64(1, "s")).astype(np.int64)


def compress_steps(x, y, c=None, close=True):
    """Keep only the points where the price (or direction) changes.

    CBMPs are often flat for many 4-second cycles, so this removes most points
    while drawing exactly the same staircase (line_shape='hv').
    Returns the start index of each run; with close=True the last index is
    appended so the final step extends to the end of the window.
    """
    n = len(y)
    if n == 0:
        return np.array([], dtype=int)
    both_nan = np.isnan(y[1:]) & np.isnan(y[:-1])
    change = (y[1:] != y[:-1]) & ~both_nan
    if c is not None:
        change |= c[1:] != c[:-1]
    idx = np.flatnonzero(np.concatenate(([True], change)))
    if close and idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def direction_segments(x, y, c):
    """Build one NaN-separated polyline per direction so that every
    horizontal step (and the riser into it) is coloured by its direction."""
    starts = compress_steps(x, y, c, close=False)
    if len(starts) == 0:
        return {}
    run_end_x = np.append(x[starts[1:]], x[-1])
    xs, ys, cs = x[starts], y[starts], c[starts]
    fin = ~np.isnan(ys)
    out = {}
    for d in (CAT_UP, CAT_DOWN, CAT_NEUTRAL):
        m = (cs == d) & fin
        hx = np.column_stack([xs[m], run_end_x[m], run_end_x[m]]).ravel()
        hy = np.column_stack([ys[m], ys[m], np.full(m.sum(), np.nan)]).ravel()
        k = np.flatnonzero((cs[1:] == d) & fin[1:] & fin[:-1]) + 1
        rx = np.column_stack([xs[k], xs[k], xs[k]]).ravel()
        ry = np.column_stack([ys[k - 1], ys[k], np.full(len(k), np.nan)]).ravel()
        out[d] = (np.concatenate([hx, rx]), np.concatenate([hy, ry]))
    return out


def y_range_control(key, arrays, default="Fit to data"):
    """Y-axis control that always fits the data actually PLOTTED in the chart.

    - Fit to data: min/max of the plotted values (+5 % margin)
    - Ignore spikes: P1–P99 of the plotted values (useful with 4-s spikes)
    - Manual: free min/max
    Returns [ymin, ymax] or None (let Plotly autoscale).
    """
    arrs = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None and len(a)]
    vals = np.concatenate(arrs) if arrs else np.array([])
    vals = vals[np.isfinite(vals)]
    options = ["Fit to data", "Ignore spikes (P1–P99)", "Manual"]
    c1, c2, c3 = st.columns([3, 1.2, 1.2])
    with c1:
        mode = st.radio(
            "Y-axis", options, index=options.index(default), horizontal=True, key=f"{key}_mode",
            help="Fit to data uses the values shown in THIS chart (e.g. the quarter-hour averages, "
                 "not the raw 4-second prices). Tip: drag on the chart to zoom, double-click to reset.",
        )
    if vals.size == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    if mode.startswith("Ignore"):
        lo, hi = (float(v) for v in np.percentile(vals, [1, 99]))
    elif mode == "Manual":
        with c2:
            lo = st.number_input("Min €/MWh", value=float(np.floor(lo)), step=10.0, key=f"{key}_min")
        with c3:
            hi = st.number_input("Max €/MWh", value=float(np.ceil(hi)), step=10.0, key=f"{key}_max")
        if hi <= lo:
            st.warning("Max must be above Min.")
            return None
        return [lo, hi]
    pad = max((hi - lo) * 0.05, 1.0)
    return [lo - pad, hi + pad]


def base_layout(fig, height, title=None):
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=15)))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=60 if title else 40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode="zoom",
    )
    fig.update_xaxes(tickformat="%H:%M", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(title_text="€/MWh", showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                     zeroline=True, zerolinecolor="rgba(128,128,128,0.4)")
    return fig


def qh_averages(times, val, cat):
    """Quarter-hour average per direction (+ overall) and share of time per direction."""
    qh = pd.DatetimeIndex(times).floor("15min")
    df = pd.DataFrame({"qh": qh, "v": val, "c": cat})
    df = df[df["c"] > 0]
    by_dir = df.groupby(["qh", "c"])["v"].mean().unstack("c")
    overall = df.groupby("qh")["v"].mean()
    idx = pd.DatetimeIndex(np.unique(qh))
    by_dir = by_dir.reindex(idx)
    res = pd.DataFrame(index=idx)
    for d in (CAT_UP, CAT_DOWN, CAT_NEUTRAL):
        res[DIR[d]["key"]] = by_dir[d] if d in by_dir.columns else np.nan
    res["overall"] = overall.reindex(idx)
    return res


# =================================================================
# SIDEBAR — GLOBAL SETTINGS
# =================================================================
today_local = datetime.now(LOCAL_TZ).date()
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    date_selected = st.date_input(
        "Date (Europe/Brussels)", value=today_local,
        min_value=date(2022, 1, 1), max_value=today_local,
    )
date_str = date_selected.strftime("%Y-%m-%d")
bucket = int(time.time() // 300) if date_selected == today_local else 0

with st.spinner(f"Loading PICASSO data for {date_str}…"):
    data = load_day(date_str, bucket)

if "error" in data:
    st.error(f"Could not load data for {date_str}: {data['error']}")
    st.stop()
if data["n"] == 0 or not data["values"]:
    st.warning("No data available for the selected date.")
    st.stop()

TIMES = data["times"]
VALUES, CATS = data["values"], data["cats"]
SOD = seconds_of_day(TIMES)
available = sorted(VALUES, key=tso_sort_key)

# TSO selection (persisted across dates, sanitised to what exists that day).
# Only write to the widget's state when something actually has to change:
# overwriting it on every run makes Streamlit drop the user's new selection.
if "tsos" not in st.session_state:
    st.session_state["tsos"] = [t for t in DEFAULT_TSOS if t in available]
else:
    _clean = [t for t in st.session_state["tsos"] if t in available]
    if _clean != list(st.session_state["tsos"]):
        st.session_state["tsos"] = _clean


def _set_tsos(lst):
    st.session_state["tsos"] = [t for t in lst if t in available]


with st.sidebar:
    t_min = dtime(0, 0)
    t_max = dtime(23, 59, 59)
    start_t, end_t = st.slider(
        "Hour range", min_value=t_min, max_value=t_max, value=(t_min, t_max),
        format="HH:mm", key="hour_range",
        help="Applies to all charts and statistics. Inside each chart you can also drag to zoom.",
    )
    st.multiselect("TSOs", options=available, format_func=tso_label, key="tsos",
                   help="The four German TSOs (50Hertz, Amprion, TransnetBW, TenneT DE) form one "
                        "LFC block and therefore share the same CBMP.")
    b1, b2, b3 = st.columns(3)
    b1.button("BE·FR·DE", on_click=_set_tsos, args=(DEFAULT_TSOS,), width="stretch")
    b2.button("CWE+", on_click=_set_tsos,
              args=(["ELIA", "RTE", "50HZT", "TNL", "APG", "SG"],), width="stretch")
    b3.button("All", on_click=_set_tsos, args=(available,), width="stretch")

    if data["empty"]:
        st.caption("In the feed but without prices on this day: "
                   + ", ".join(tso_label(c) for c in data["empty"]))
    st.divider()
    st.markdown(
        f"**Read more on Gem Energy Analytics**\n\n"
        f"- {post_link('picasso')}\n- {post_link('overlap')}\n- {post_link('overlap2')}\n"
        f"- {post_link('imbalance')}\n- {post_link('intro')}"
    )
    st.markdown(
        "[Subscribe](https://gemenergyanalytics.substack.com/) · "
        "[LinkedIn](https://www.linkedin.com/in/julien-jomaux/) · "
        "[Email](mailto:julien.jomaux@gmail.com)"
    )

SELECTED = list(st.session_state["tsos"])
MASK = (SOD >= start_t.hour * 3600 + start_t.minute * 60 + start_t.second) & (
    SOD <= end_t.hour * 3600 + end_t.minute * 60 + end_t.second)
T = TIMES[MASK]


def color_of(code):
    if code in FIXED_COLORS:
        return FIXED_COLORS[code]
    return PALETTE[(tso_sort_key(code)[0] + 3) % len(PALETTE)]


# =================================================================
# HEADER
# =================================================================
st.markdown(
    f"""
<div class="hero">
  <h1>⚡ PICASSO aFRR prices · {date_selected.strftime('%A %d %B %Y')}</h1>
  <p>Cross-Border Marginal Prices (CBMP) of the European aFRR platform, every 4 seconds.
  Data: <a href="https://www.transnetbw.de/en/energy-market/ancillary-services/picasso" target="_blank">TransnetBW</a>
  · Analysis: <a href="https://gemenergyanalytics.substack.com/" target="_blank">Gem Energy Analytics</a>
  by <a href="https://www.linkedin.com/in/julien-jomaux/" target="_blank">Julien Jomaux</a></p>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("📖 What am I looking at? PICASSO and CBMP in 1 minute", expanded=False):
    st.markdown(
        f"""
**PICASSO** is the European platform for the exchange of **aFRR balancing energy** (automatic
Frequency Restoration Reserve). Every **4 seconds**, it collects the aFRR needs of all connected
TSOs, nets opposite needs across borders, and activates the cheapest bids from a common merit
order, as long as cross-border capacity allows.

The resulting price is the **CBMP (Cross-Border Marginal Price)**: one price per TSO and per
4-second optimisation cycle (225 per quarter-hour). When there is no congestion, neighbouring
TSOs share the same price; when capacity is saturated, prices split.

The feed gives two prices per TSO:
- **POS** (up): price of upward aFRR activation → shown as <span class="pill" style="background:{DIR[CAT_UP]['color']}">Up</span>
- **NEG** (down): price of downward aFRR activation → shown as <span class="pill" style="background:{DIR[CAT_DOWN]['color']}">Down</span>
- If both prices are published in the same cycle, it means **no aFRR activation** took place in that cycle → shown as <span class="pill" style="background:{DIR[CAT_NEUTRAL]['color']}">No activation</span> (the line is drawn at the midpoint of the two prices)

Prices are paid **pay-as-clear**: all activated bids receive the CBMP. These prices feed directly
into imbalance prices in many countries (e.g. Belgium).

**Go deeper:** {post_link('picasso')} · {post_link('reserves')} · {post_link('imbalance')} · {post_link('intro')}
""",
        unsafe_allow_html=True,
    )

if not SELECTED:
    st.info("👈 Select at least one TSO in the sidebar.")
    st.stop()

missing = [t for t in SELECTED if not np.isfinite(VALUES[t][MASK]).any()]
if missing:
    st.warning("No prices in the selected hour range for: " + ", ".join(tso_label(t) for t in missing))

# =================================================================
# DAY AT A GLANCE
# =================================================================
st.subheader("Day at a glance")
rows = []
for t in SELECTED:
    v, c = VALUES[t][MASK], CATS[t][MASK]
    ok = c > 0
    tot = max(int(ok.sum()), 1)
    up_v, dn_v = v[c == CAT_UP], v[c == CAT_DOWN]
    rows.append({
        "TSO": tso_label(t),
        "Avg up price": np.nanmean(up_v) if up_v.size else np.nan,
        "Avg down price": np.nanmean(dn_v) if dn_v.size else np.nan,
        "Min": np.nanmin(v) if ok.any() else np.nan,
        "Max": np.nanmax(v) if ok.any() else np.nan,
        "% time Up": 100 * np.sum(c == CAT_UP) / tot,
        "% time Down": 100 * np.sum(c == CAT_DOWN) / tot,
        "% time no activation": 100 * np.sum(c == CAT_NEUTRAL) / tot,
    })
glance = pd.DataFrame(rows).set_index("TSO")
st.dataframe(
    glance,
    width="stretch",
    column_config={
        "Avg up price": st.column_config.NumberColumn(format="%.1f €/MWh"),
        "Avg down price": st.column_config.NumberColumn(format="%.1f €/MWh"),
        "Min": st.column_config.NumberColumn(format="%.1f"),
        "Max": st.column_config.NumberColumn(format="%.1f"),
        "% time Up": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "% time Down": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "% time no activation": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
    },
)
st.caption("Averages are time-weighted over 4-second cycles, per direction. "
           "% time is the share of 4-second cycles with an up activation, a down activation, or no activation (both prices published).")


# =================================================================
# SECTION 1 — COMBINED CHART
# =================================================================
def section_combined():
    st.header("All selected TSOs — combined")
    with st.expander("⚙️ Chart controls", expanded=False):
        yr = y_range_control("s1", [VALUES[t][MASK] for t in SELECTED])
    fig = go.Figure()
    for t in SELECTED:
        y = VALUES[t][MASK]
        idx = compress_steps(T, y)
        fig.add_trace(go.Scattergl(
            x=T[idx], y=y[idx], mode="lines", name=tso_label(t),
            line=dict(color=color_of(t), width=1.6, shape="hv"),
            hovertemplate="%{y:.2f} €/MWh",
        ))
    base_layout(fig, 520, f"CBMP {date_str} · {start_t:%H:%M}–{end_t:%H:%M} (Europe/Brussels)")
    if yr:
        fig.update_yaxes(range=yr)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    st.caption("Drag to zoom · double-click to reset · click a legend entry to hide a TSO, "
               "double-click it to isolate. Use the slider below the chart to pan through the day.")


section_combined()


# =================================================================
# SECTION 2 — INDIVIDUAL TSO CHARTS COLOURED BY DIRECTION
# =================================================================
def section_individual():
    st.header("Individual TSOs — coloured by aFRR direction")
    with st.expander("⚙️ Chart controls", expanded=False):
        shared_y = st.checkbox("Same Y-scale for all TSOs", value=True, key="s2_shared")
        yr = y_range_control("s2", [VALUES[t][MASK] for t in SELECTED]) if shared_y else None
    n = len(SELECTED)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[tso_label(t) for t in SELECTED],
                        vertical_spacing=min(0.06, 0.25 / max(n, 1)))
    for i, t in enumerate(SELECTED, start=1):
        segs = direction_segments(T, VALUES[t][MASK], CATS[t][MASK])
        for d, (sx, sy) in segs.items():
            fig.add_trace(go.Scattergl(
                x=sx, y=sy, mode="lines", name=DIR[d]["desc"], legendgroup=DIR[d]["key"],
                showlegend=(i == 1), line=dict(color=DIR[d]["color"], width=1.4),
                connectgaps=False, hovertemplate="%{y:.2f} €/MWh",
            ), row=i, col=1)
    base_layout(fig, 90 + 240 * n)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(title_text=None)
    if yr:
        fig.update_yaxes(range=yr)
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    st.caption("X-axes are linked: zooming in one chart zooms all of them.")


section_individual()


# =================================================================
# SECTION 3 — QUARTER-HOUR AVERAGES BY DIRECTION
# =================================================================
def section_qh():
    st.header("Quarter-hour average prices by direction")
    st.markdown(
        f"Within one quarter-hour, PICASSO can clear 225 times, in both directions. When the "
        f"common up and down merit orders **overlap**, the average *down* price can end up "
        f"**above** the average *up* price in the same quarter-hour, as if a surplus looked like "
        f"a shortage. Explained in {post_link('overlap')} and {post_link('overlap2')}."
    )
    labels = {"up": "Up", "down": "Down", "neutral": "No activation (midpoint)", "overall": "All cycles (overall avg)"}
    colors = {"up": DIR[CAT_UP]["color"], "down": DIR[CAT_DOWN]["color"],
              "neutral": DIR[CAT_NEUTRAL]["color"], "overall": "#7f8c8d"}

    qh = {t: qh_averages(T, VALUES[t][MASK], CATS[t][MASK]) for t in SELECTED}

    with st.expander("⚙️ Chart controls", expanded=True):
        dirs = st.multiselect("Series", options=list(labels), default=["up", "down"],
                              format_func=labels.get, key="s3_dirs")
        yr = y_range_control("s3", [qh[t][d].to_numpy() for t in SELECTED for d in dirs])

    if not dirs:
        st.info("Select at least one series.")
        return

    n = len(SELECTED)
    ncols = 1 if n == 1 else 2
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=[tso_label(t) for t in SELECTED],
                        vertical_spacing=min(0.12, 0.3 / max(nrows, 1)), horizontal_spacing=0.04)
    for i, t in enumerate(SELECTED):
        r, c = i // ncols + 1, i % ncols + 1
        for d in dirs:
            s = qh[t][d]
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines+markers", name=labels[d], legendgroup=d,
                showlegend=(i == 0),
                line=dict(color=colors[d], width=1.6, shape="hv",
                          dash="dot" if d == "overall" else "solid"),
                marker=dict(size=4), connectgaps=False,
                hovertemplate=labels[d] + ": %{y:.2f} €/MWh<extra></extra>",
            ), row=r, col=c)
    base_layout(fig, 70 + 320 * nrows)
    fig.update_yaxes(title_text=None)
    if yr:
        fig.update_yaxes(range=yr)
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)

    # ---- statistics --------------------------------------------------
    st.subheader("Quarter-hour direction statistics")
    diff_rows, count_rows = [], []
    for t in SELECTED:
        a = qh[t]
        has_up, has_dn, has_nt = a["up"].notna(), a["down"].notna(), a["neutral"].notna()
        diff = (a["up"] - a["down"]).dropna()
        diff_rows.append({
            "TSO": tso_label(t),
            "QH with Up & Down": int(len(diff)),
            "Down > Up (overlap)": int((diff < 0).sum()),
            "% overlap": 100 * (diff < 0).mean() if len(diff) else np.nan,
            "Mean": diff.mean() if len(diff) else np.nan,
            "Median": diff.median() if len(diff) else np.nan,
            "P10": diff.quantile(.10) if len(diff) else np.nan,
            "P90": diff.quantile(.90) if len(diff) else np.nan,
        })
        count_rows.append({
            "TSO": tso_label(t),
            "Both Up & Down": int((has_up & has_dn).sum()),
            "Only Up (± no activation)": int((has_up & ~has_dn).sum()),
            "Only Down (± no activation)": int((has_dn & ~has_up).sum()),
            "No activation at all": int((has_nt & ~has_up & ~has_dn).sum()),
            "Total QH": int(len(a)),
        })

    st.markdown("**Up − Down spread of quarter-hour averages** (€/MWh, only quarter-hours where both "
                "directions occurred). A negative spread means the average down price was higher "
                "than the average up price: the signature of overlapping merit orders.")
    st.dataframe(
        pd.DataFrame(diff_rows).set_index("TSO"), width="stretch",
        column_config={
            "% overlap": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
            **{k: st.column_config.NumberColumn(format="%.1f") for k in ["Mean", "Median", "P10", "P90"]},
        },
    )
    st.markdown("**Number of quarter-hours by direction combination**")
    st.dataframe(pd.DataFrame(count_rows).set_index("TSO"), width="stretch")

    export = pd.concat({tso_label(t): qh[t] for t in SELECTED}, axis=1)
    export.index.name = "QH start (Europe/Brussels)"
    st.download_button(
        "⬇️ Download quarter-hour averages (CSV)",
        export.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
        file_name=f"picasso_qh_averages_{date_str}.csv", mime="text/csv",
    )


section_qh()


# =================================================================
# SECTION 4 — SIMILARITY MATRIX
# =================================================================
def section_similarity():
    st.header("Price coupling between TSOs")
    st.caption("Share (%) of 4-second cycles where two TSOs have exactly the same CBMP, among cycles "
               "where both have a price. 100 % = always coupled (no congestion between them).")
    with st.expander("⚙️ Chart controls", expanded=False):
        use_all = st.checkbox("Use all TSOs (not only the selection)", value=False, key="s4_all")
    tsos = available if use_all else SELECTED
    if len(tsos) < 2:
        st.info("Select at least two TSOs.")
        return
    M = np.vstack([VALUES[t][MASK] for t in tsos])
    fin = np.isfinite(M)
    m = len(tsos)
    sim = np.full((m, m), np.nan)
    for i in range(m):
        both = fin[i] & fin
        eq = (M[i] == M) & both
        cnt = both.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sim[i] = np.where(cnt > 0, 100 * eq.sum(axis=1) / cnt, np.nan)
    labels = [tso_label(t) for t in tsos]
    fig = go.Figure(go.Heatmap(
        z=sim, x=labels, y=labels, colorscale="RdYlGn", zmin=0, zmax=100,
        text=np.where(np.isnan(sim), "", np.vectorize(lambda v: f"{v:.0f}")(np.nan_to_num(sim))),
        texttemplate="%{text}", hovertemplate="%{y} ↔ %{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="%", thickness=12),
    ))
    fig.update_layout(height=max(380, 34 * m + 160), margin=dict(l=10, r=10, t=20, b=10))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


section_similarity()

# =================================================================
# FOOTER
# =================================================================
st.divider()
st.markdown(
    f"""
<div style="text-align:center; opacity:.8; font-size:.9rem">
Built by <b>Julien Jomaux</b> · <a href="https://gemenergyanalytics.substack.com/" target="_blank">Gem Energy Analytics</a>
· Data: TransnetBW PICASSO CBMP API · Times in Europe/Brussels
</div>
""",
    unsafe_allow_html=True,
)
