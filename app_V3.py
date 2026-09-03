import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import io
import urllib3
from zoneinfo import ZoneInfo
import itertools

st.set_page_config(page_title="Picasso aFRR prices - Visualizer", layout="wide")

# ---------------------------------------------------------------
# SETTINGS UI (top of page)
# ---------------------------------------------------------------
LOCAL_TZ = ZoneInfo("Europe/Brussels")
date_selected = st.date_input(
    "Select a date (Europe/Brussels)",
    value=datetime.now(LOCAL_TZ).date(),
    min_value=date(2020, 1, 1)
)

date_str = date_selected.strftime("%Y-%m-%d")
st.title(f"Picasso aFRR prices for {date_str} ")

# Description and data source
# -------------------------------
st.markdown(
    """
This app presents the aFRR prices (CBMP for Cross-border Marginal Prices) from the Picasso platform. aFRR prices vary every 4 seconds.

**Data source:** [Transnet](https://www.transnetbw.de/en/energy-market/ancillary-services/picasso)

**More insights:** [GEM Energy Analytics](https://gemenergyanalytics.substack.com/)
**Connect with me:** [Julien Jomaux](https://www.linkedin.com/in/julien-jomaux/)
**Email me:** [julien.jomaux@gmail.com](mailto:julien.jomaux@gmail.com)
"""
)


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)




# ---------------------------------------------------------------
# TSO definitions (includes AMP, TNG, TTG; CEPS corrected)
# ---------------------------------------------------------------
TSO_DISPLAY_NAMES = {
    "50HZT":  "50HZT (Germany)",
    "APG":    "APG (Austria)",
    "ELIA":   "Elia (Belgium)",
    "TNL":    "TNL (Netherlands)",
    "RTE":    "RTE (France)",
    "CEPS":   "CEPS (Czechia)",
    "TERNA":  "TERNA (Italy)",
    "ESO":    "ESO (Bulgaria)",
    "ENDK1":  "ENDK1 (Denmark 1)",
    "ENDK2":  "ENDK2 (Denmark 2)",
    "SEPS":   "SEPS (Slovakia)",
    "LITGRID":"LITGRID (Lithuania)",
    "ADMIE":  "ADMIE (Greece)",
    "FINGRID":"FINGRID (Finland)",
    "ELERING":"ELERING (Estonia)",
    "AST":    "AST (Latvia)",
    "REE":    "REE (Spain)",
    "PSE":    "PSE (Poland)",
    "AMP":    "AMP (Germany)",
    "TNG":    "TNG (Germany)",
    "TTG":    "TTG (Germany)"
}

# Color cycle for all TSOs
COLOR_CYCLE = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
TSO_COLORS = {tso: next(COLOR_CYCLE) for tso in TSO_DISPLAY_NAMES.keys()}

# Colors used to encode the aFRR direction of a price point.
# - "up"      -> only a POS (upward) price is available for that timestamp -> aFRR up bid only
# - "down"    -> only a NEG (downward) price is available for that timestamp -> aFRR down bid only
# - "neutral" -> both POS and NEG prices are available -> no net aFRR demand
DIRECTION_COLORS = {"up": "red", "down": "green", "neutral": "blue"}
DIRECTION_LABELS = {"up": "Up", "down": "Down", "neutral": "Neutral"}




# ---------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def load_csv_for_date(date_str: str):
    url = f"https://api.transnetbw.de/picasso-cbmp/csv?date={date_str}&lang=de"
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        st.error(f"API error {response.status_code}")
        return None
    df = pd.read_csv(io.StringIO(response.content.decode()), sep=";", parse_dates=["Zeit (ISO 8601)"])
    df["Zeit (ISO 8601)"] = pd.to_datetime(df["Zeit (ISO 8601)"], utc=True).dt.tz_convert(LOCAL_TZ)
    return df

df_raw = load_csv_for_date(date_str)

if df_raw is None or df_raw.empty:
    st.warning("No data available for the selected date.")
    st.stop()

df_raw = df_raw.reset_index(drop=True)
times_all = df_raw["Zeit (ISO 8601)"]

# ---------------------------------------------------------------
# PREPARE TSO DATA (computed once, for the full day)
# ---------------------------------------------------------------
# For each TSO we keep two parallel arrays (aligned with df_raw / times_all):
#   tso_values_all[tso]   -> the merged CBMP price (as before)
#   tso_category_all[tso] -> the direction of that price point:
#                             "up"      -> only the POS (up) price exists     -> aFRR up bid only
#                             "down"    -> only the NEG (down) price exists   -> aFRR down bid only
#                             "neutral" -> both POS and NEG prices exist      -> no net aFRR demand
#                             None      -> neither price exists (no data)
tso_values_all = {}
tso_category_all = {}
for tso in TSO_DISPLAY_NAMES.keys():
    neg_col = f"{tso}_NEG"   # down price
    pos_col = f"{tso}_POS"   # up price
    if neg_col in df_raw.columns or pos_col in df_raw.columns:
        vals = []
        cats = []
        for neg, pos in zip(df_raw.get(neg_col, [np.nan]*len(df_raw)), df_raw.get(pos_col, [np.nan]*len(df_raw))):
            neg_val = np.nan if pd.isna(neg) or neg == "N/A" else float(neg)
            pos_val = np.nan if pd.isna(pos) or pos == "N/A" else float(pos)
            neg_present = not np.isnan(neg_val)
            pos_present = not np.isnan(pos_val)
            if not neg_present and not pos_present:
                vals.append(np.nan)
                cats.append(None)
            elif neg_present and not pos_present:
                vals.append(neg_val)
                cats.append("down")
            elif pos_present and not neg_present:
                vals.append(pos_val)
                cats.append("up")
            else:
                vals.append((neg_val + pos_val) / 2)
                cats.append("neutral")
        tso_values_all[tso] = np.array(vals)
        tso_category_all[tso] = np.array(cats, dtype=object)

available_tsos = [tso for tso in TSO_DISPLAY_NAMES.keys() if tso in tso_values_all]

if not available_tsos:
    st.warning("No TSO price columns found for the selected date.")
    st.stop()

_ALL_LOCAL_TIMES = times_all.dt.time
DAY_MIN_TIME, DAY_MAX_TIME = min(_ALL_LOCAL_TIMES), max(_ALL_LOCAL_TIMES)

# ---------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------
def time_range_widget(key):
    """Independent 'Select Hour Range' slider for one section."""
    return st.slider(
        "Select Hour Range",
        min_value=DAY_MIN_TIME,
        max_value=DAY_MAX_TIME,
        value=(DAY_MIN_TIME, DAY_MAX_TIME),
        format="HH:mm",
        key=key,
    )


def time_mask(start_t, end_t):
    return ((times_all.dt.time >= start_t) & (times_all.dt.time <= end_t)).to_numpy()


def tso_widget(key, default_n=6):
    return st.multiselect(
        "Choose TSOs",
        options=available_tsos,
        format_func=lambda x: TSO_DISPLAY_NAMES[x],
        default=available_tsos[:default_n],
        key=key,
    )


def yaxis_widget(key, selected, mask, values_source=None):
    """Y-axis range slider based on the valid values of `selected` TSOs
    within `mask`. Returns (ymin, ymax) or (None, None) if no data.

    Includes a "Fit Y-axis to data" button that snaps the slider tightly
    around the actual min/max of the currently selected data (with a small
    visual margin), so all points are visible as clearly as possible.
    """
    source = values_source if values_source is not None else tso_values_all
    if not selected:
        return None, None
    arrs = [source[t][mask] for t in selected if t in source]
    if not arrs:
        return None, None
    all_vals = np.concatenate(arrs)
    valid_vals = all_vals[~np.isnan(all_vals)]
    if valid_vals.size == 0:
        return None, None

    data_min = float(valid_vals.min())
    data_max = float(valid_vals.max())

    # Wide bounds for the slider itself, so the user can still zoom out.
    dmin = float(np.floor(data_min - 20))
    dmax = float(np.ceil(data_max + 20))

    # Tight "fit to data" target: actual min/max plus a small visual margin
    # (so points don't sit glued to the plot border).
    margin = max((data_max - data_min) * 0.05, 0.5)
    fit_min = max(float(np.floor((data_min - margin) * 10) / 10), dmin)
    fit_max = min(float(np.ceil((data_max + margin) * 10) / 10), dmax)

    # If a value is already stored for this widget (from a previous run or a
    # previous, different TSO/time selection), keep it but clamp it back
    # inside the current bounds so the slider never errors out.
    if key in st.session_state:
        lo, hi = st.session_state[key]
        lo = min(max(lo, dmin), dmax)
        hi = min(max(hi, dmin), dmax)
        if hi < lo:
            lo, hi = dmin, dmax
        st.session_state[key] = (lo, hi)

    if st.button("🔍 Fit Y-axis to data", key=f"{key}_fit_btn",
                 help="Snap the Y-axis range to tightly fit the currently selected data"):
        st.session_state[key] = (fit_min, fit_max)

    slider_kwargs = dict(min_value=dmin, max_value=dmax, step=0.5, key=key)
    if key not in st.session_state:
        slider_kwargs["value"] = (dmin, dmax)

    return st.slider("Select Y-axis range (€/MWh)", **slider_kwargs)


def plot_direction_staircase(ax_i, times_series, values, categories,
                              color_map=DIRECTION_COLORS, linewidth=1.5):
    """
    Draw a staircase (steps-post) line on ax_i where every horizontal and
    vertical segment is colored according to the aFRR direction of the
    price it belongs to:
        red   -> up only
        green -> down only
        blue  -> neutral (both directions)
    Segments touching a missing value (no data) are simply not drawn,
    leaving a gap.
    """
    # Invisible plot first: this makes matplotlib treat the x-axis as a
    # proper date axis and sets sensible autoscale limits, without
    # actually drawing a (single-colored) line.
    ax_i.plot(times_series, values, alpha=0)

    x = mdates.date2num(np.array(times_series.dt.to_pydatetime()))
    y = np.asarray(values, dtype=float)
    cats = list(categories)
    n = len(x)

    segments = []
    seg_colors = []
    for i in range(n - 1):
        if np.isnan(y[i]) or np.isnan(y[i + 1]) or cats[i] is None or cats[i + 1] is None:
            continue
        # Horizontal step: value y[i] is held from x[i] to x[i+1]
        segments.append([(x[i], y[i]), (x[i + 1], y[i])])
        seg_colors.append(color_map.get(cats[i], "gray"))
        # Vertical riser: jump from y[i] to y[i+1] happening at x[i+1]
        segments.append([(x[i + 1], y[i]), (x[i + 1], y[i + 1])])
        seg_colors.append(color_map.get(cats[i + 1], "gray"))

    if segments:
        lc = LineCollection(segments, colors=seg_colors, linewidths=linewidth)
        ax_i.add_collection(lc)


DIRECTION_LEGEND_ELEMENTS = [
    Line2D([0], [0], color="red", lw=2, label="Up only (aFRR up bid)"),
    Line2D([0], [0], color="green", lw=2, label="Down only (aFRR down bid)"),
    Line2D([0], [0], color="blue", lw=2, label="Neutral (both directions)"),
]


def qh_direction_averages(values, categories, qh_bucket_series, qh_idx):
    """
    Returns a dict with one pandas Series per direction ("up", "down",
    "neutral"), indexed by qh_idx, containing the average price of that
    direction for each quarter-hour (NaN where that direction did not
    occur in that quarter-hour).
    """
    s = pd.DataFrame({
        "qh": qh_bucket_series,
        "value": np.asarray(values, dtype=float),
        "cat": np.asarray(categories, dtype=object),
    })
    out = {}
    for direction in ("up", "down", "neutral"):
        sub = s[s["cat"] == direction]
        avg = sub.groupby("qh")["value"].mean()
        out[direction] = avg.reindex(qh_idx)
    return out


def qh_direction_combo_counts(has_up, has_down, has_neutral):
    """
    Counts quarter-hours into 4 mutually-exclusive buckets based on which
    directions had at least one price point in it:
      - "Both Up & Down"           -> an Up average AND a Down average exist
                                       (neutral may or may not also be present)
      - "Only Up (± Neutral)"      -> an Up average exists, no Down average
                                       (neutral may or may not also be present)
      - "Only Down (± Neutral)"    -> a Down average exists, no Up average
                                       (neutral may or may not also be present)
      - "Only Neutral"             -> only a Neutral average exists
    Quarter-hours with no data at all in any direction are not counted in
    any of the 4 buckets (their sum can be less than the total QH count).
    """
    has_up = np.asarray(has_up, dtype=bool)
    has_down = np.asarray(has_down, dtype=bool)
    has_neutral = np.asarray(has_neutral, dtype=bool)

    both = int(np.sum(has_up & has_down))
    only_up = int(np.sum(has_up & ~has_down))
    only_down = int(np.sum(has_down & ~has_up))
    only_neutral = int(np.sum(has_neutral & ~has_up & ~has_down))

    return {
        "Both Up & Down": both,
        "Only Up (± Neutral)": only_up,
        "Only Down (± Neutral)": only_down,
        "Only Neutral": only_neutral,
    }


QH_CATEGORY_ORDER = ["Both Up & Down", "Only Up (± Neutral)", "Only Down (± Neutral)", "Only Neutral"]


# =================================================================
# SECTION 1 — ALL TSOs COMBINED CHART
# =================================================================
st.header("All TSOs — Combined Chart")

with st.expander("⚙️ Chart controls", expanded=True):
    start1, end1 = time_range_widget("sec1_time")
    tsos1 = tso_widget("sec1_tso")
    mask1 = time_mask(start1, end1)
    ymin1, ymax1 = yaxis_widget("sec1_yaxis", tsos1, mask1)

if not tsos1:
    st.warning("Please select at least one TSO to display the combined chart.")
elif ymin1 is None:
    st.warning("No valid numerical data for the selected TSOs / time range.")
else:
    times1 = times_all[mask1]
    fig, ax = plt.subplots(figsize=(18, 7))
    for tso in tsos1:
        ax.plot(times1, tso_values_all[tso][mask1], label=TSO_DISPLAY_NAMES[tso], color=TSO_COLORS[tso])
    ax.set_title(f"{date_str} Picasso CBMP (Local: {start1.strftime('%H:%M')} – {end1.strftime('%H:%M')})")
    ax.set_xlabel("Time (Europe/Brussels)")
    ax.set_ylabel("€/MWh")
    ax.set_ylim(ymin1, ymax1)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))
    ax.grid(True)
    ax.legend(ncols=2)
    plt.tight_layout()
    st.pyplot(fig)

# =================================================================
# SECTION 2 — INDIVIDUAL TSO GRAPHS (colored by aFRR direction)
# =================================================================
st.header("Individual TSO Graphs")

with st.expander("⚙️ Chart controls", expanded=True):
    start2, end2 = time_range_widget("sec2_time")
    tsos2 = tso_widget("sec2_tso")
    mask2 = time_mask(start2, end2)
    ymin2, ymax2 = yaxis_widget("sec2_yaxis", tsos2, mask2)

st.caption("Red = aFRR up bid only · Green = aFRR down bid only · Blue = neutral (both directions present)")

if not tsos2:
    st.warning("Please select at least one TSO to display the individual graphs.")
elif ymin2 is None:
    st.warning("No valid numerical data for the selected TSOs / time range.")
else:
    times2 = times_all[mask2]
    n2 = len(tsos2)
    rows2 = int(np.ceil(n2 / 2))
    fig2, axs2 = plt.subplots(rows2, 2, figsize=(20, 4 * rows2), sharex=True, sharey=True)
    axs2 = np.atleast_1d(axs2).flatten()

    for i, tso in enumerate(tsos2):
        ax_i = axs2[i]
        plot_direction_staircase(ax_i, times2, tso_values_all[tso][mask2], tso_category_all[tso][mask2])
        ax_i.set_title(TSO_DISPLAY_NAMES[tso])
        ax_i.set_ylim(ymin2, ymax2)
        ax_i.grid(True)
        ax_i.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))

    for j in range(i + 1, len(axs2)):
        axs2[j].axis("off")

    fig2.legend(handles=DIRECTION_LEGEND_ELEMENTS, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout()
    st.pyplot(fig2)

# =================================================================
# SECTION 3 — QUARTER-HOUR AVERAGE PRICES BY DIRECTION (interactive)
# =================================================================
st.header("Quarter-Hour Average Prices by Direction")

with st.expander("⚙️ Chart controls", expanded=True):
    start3, end3 = time_range_widget("sec3_time")
    tsos3 = tso_widget("sec3_tso")
    mask3 = time_mask(start3, end3)
    ymin3, ymax3 = yaxis_widget("sec3_yaxis", tsos3, mask3)
    directions3 = st.multiselect(
        "Directions to show",
        options=["up", "down", "neutral"],
        default=["up", "down", "neutral"],
        format_func=lambda d: DIRECTION_LABELS[d],
        key="sec3_directions",
    )

st.caption(
    "For every 15-minute block, the average price is computed separately for the up, down and "
    "neutral directions (Red = Up, Green = Down, Blue = Neutral). If a direction has no data in a "
    "given quarter-hour, that point is left blank. Hover, zoom and click a legend entry to "
    "show/hide a series."
)

qh_avgs_by_tso = {}
qh_index3 = None

if not tsos3:
    st.warning("Please select at least one TSO to display the quarter-hour chart.")
elif not directions3:
    st.warning("Please select at least one direction (Up / Down / Neutral) to display.")
elif ymin3 is None:
    st.warning("No valid numerical data for the selected TSOs / time range.")
else:
    times3 = times_all[mask3]
    qh_bucket3 = times3.dt.floor("15min").reset_index(drop=True)
    qh_index3 = pd.DatetimeIndex(sorted(qh_bucket3.unique()))
    # Plotly does not need a tz-aware index to render local wall-clock time correctly here,
    # since qh_index3 already carries Europe/Brussels local time; strip tz to avoid Plotly
    # re-interpreting/re-converting the axis.
    qh_index3_naive = qh_index3.tz_localize(None)

    n3 = len(tsos3)
    rows3 = int(np.ceil(n3 / 2))
    fig3 = make_subplots(
        rows=rows3, cols=2,
        subplot_titles=[TSO_DISPLAY_NAMES[t] for t in tsos3],
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.12 / max(rows3, 1),
    )

    legend_shown = {"up": False, "down": False, "neutral": False}
    for i, tso in enumerate(tsos3):
        r = i // 2 + 1
        c = i % 2 + 1
        qh_avgs = qh_direction_averages(
            tso_values_all[tso][mask3], tso_category_all[tso][mask3], qh_bucket3, qh_index3
        )
        qh_avgs_by_tso[tso] = qh_avgs
        for direction in ("up", "down", "neutral"):
            if direction not in directions3:
                continue
            fig3.add_trace(
                go.Scatter(
                    x=qh_index3_naive,
                    y=qh_avgs[direction].values,
                    mode="lines+markers",
                    name=DIRECTION_LABELS[direction],
                    legendgroup=direction,
                    showlegend=not legend_shown[direction],
                    line=dict(color=DIRECTION_COLORS[direction], width=1.5),
                    marker=dict(size=4),
                    connectgaps=False,
                    hovertemplate="%{x|%H:%M} · " + DIRECTION_LABELS[direction] + ": %{y:.2f} €/MWh<extra></extra>",
                ),
                row=r, col=c,
            )
            legend_shown[direction] = True

    fig3.update_yaxes(range=[ymin3, ymax3])
    fig3.update_xaxes(tickformat="%H:%M")
    fig3.update_layout(
        height=350 * rows3,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80),
    )
    st.plotly_chart(fig3, width="stretch")

# -----------------------------------------------------------------
# SECTION 3b — QH DIRECTION STATISTICS
# -----------------------------------------------------------------
if tsos3 and directions3 and qh_index3 is not None and len(qh_index3) > 0:
    st.subheader("Quarter-Hour Direction Statistics")

    diff_rows = []
    count_rows = []
    for tso in tsos3:
        qh_avgs = qh_avgs_by_tso[tso]
        has_up = qh_avgs["up"].notna()
        has_down = qh_avgs["down"].notna()
        has_neutral = qh_avgs["neutral"].notna()

        # --- Up minus Down stats, only for QH where BOTH exist ---
        diff = (qh_avgs["up"] - qh_avgs["down"]).dropna()
        if len(diff) > 0:
            diff_rows.append({
                "TSO": TSO_DISPLAY_NAMES[tso],
                "N (QH with Up & Down)": int(len(diff)),
                "Mean (€/MWh)": diff.mean(),
                "Median (€/MWh)": diff.median(),
                "Std (€/MWh)": diff.std(),
                "P10": diff.quantile(0.10),
                "P25": diff.quantile(0.25),
                "P75": diff.quantile(0.75),
                "P90": diff.quantile(0.90),
            })
        else:
            diff_rows.append({
                "TSO": TSO_DISPLAY_NAMES[tso],
                "N (QH with Up & Down)": 0,
                "Mean (€/MWh)": np.nan,
                "Median (€/MWh)": np.nan,
                "Std (€/MWh)": np.nan,
                "P10": np.nan,
                "P25": np.nan,
                "P75": np.nan,
                "P90": np.nan,
            })

        # --- Count of QH per direction combination ---
        combo_counts = qh_direction_combo_counts(has_up.values, has_down.values, has_neutral.values)
        row = {"TSO": TSO_DISPLAY_NAMES[tso]}
        for cat in QH_CATEGORY_ORDER:
            row[cat] = combo_counts[cat]
        row["Total QH"] = int(len(qh_index3))
        count_rows.append(row)

    diff_df = pd.DataFrame(diff_rows).set_index("TSO")
    st.markdown("**Up − Down spread** (computed only for quarter-hours where both an Up and a Down average exist)")
    st.dataframe(diff_df.style.format("{:.2f}", na_rep="—"), width="stretch")

    count_df = pd.DataFrame(count_rows).set_index("TSO")
    st.markdown(
        "**Number of quarter-hours by direction combination** "
        "(\"± Neutral\" means neutral prices may or may not also be present in that quarter-hour; "
        "the 4 columns can sum to less than Total QH if some quarter-hours have no data at all)"
    )
    st.dataframe(count_df, width="stretch")

# =================================================================
# SECTION 4 — TSO SIMILARITY MATRIX
# =================================================================
st.header("TSO Similarity Matrix (%)")

with st.expander("⚙️ Chart controls", expanded=True):
    start4, end4 = time_range_widget("sec4_time")
    tsos4 = tso_widget("sec4_tso")
    mask4 = time_mask(start4, end4)
    ymin4, ymax4 = yaxis_widget("sec4_yaxis", tsos4, mask4)

st.caption(
    "The Y-axis range restricts the comparison to price points within that range: values outside "
    "it are excluded (treated as missing) before computing the % of matching timestamps."
)


def percentage_equal(arr1: np.ndarray, arr2: np.ndarray) -> float:
    mask = ~np.isnan(arr1) & ~np.isnan(arr2)
    if mask.sum() == 0:
        return np.nan
    return 100.0 * np.sum(arr1[mask] == arr2[mask]) / mask.sum()


def restrict_to_yrange(values, ymin, ymax):
    v = values.astype(float).copy()
    if ymin is not None and ymax is not None:
        out_of_range = (v < ymin) | (v > ymax)
        v[out_of_range] = np.nan
    return v


if not tsos4:
    st.warning("Please select at least one TSO to display the similarity matrix.")
elif ymin4 is None:
    st.warning("No valid numerical data for the selected TSOs / time range.")
else:
    vals4 = {
        tso: restrict_to_yrange(tso_values_all[tso][mask4], ymin4, ymax4)
        for tso in tsos4
    }
    labels4 = [TSO_DISPLAY_NAMES[t] for t in tsos4]
    m4 = len(tsos4)
    sim_matrix = np.full((m4, m4), np.nan, dtype=float)

    for i in range(m4):
        for j in range(m4):
            if i == j:
                sim_matrix[i, j] = 100.0
            else:
                sim_matrix[i, j] = percentage_equal(vals4[tsos4[i]], vals4[tsos4[j]])

    sim_df = pd.DataFrame(sim_matrix, index=labels4, columns=labels4)

    styled = (
        sim_df.style
        .background_gradient(cmap="RdYlGn", vmin=0, vmax=100, axis=None)
        .format(lambda v: f"{v:.2f}%" if not pd.isna(v) else "")
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    )

    st.dataframe(styled, width="stretch")
