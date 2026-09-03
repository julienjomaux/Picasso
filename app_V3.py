import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
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

# ---------------------------------------------------------------
# TIME RANGE OPTIONS
# ---------------------------------------------------------------
local_times = df_raw["Zeit (ISO 8601)"].dt.time
min_local_time, max_local_time = min(local_times), max(local_times)

st.subheader("Time Range")
start_time, end_time = st.slider(
    "Select Hour Range",
    min_value=min_local_time,
    max_value=max_local_time,
    value=(min_local_time, max_local_time),
    format="HH:mm"
)

df = df_raw[
    (df_raw["Zeit (ISO 8601)"].dt.time >= start_time) &
    (df_raw["Zeit (ISO 8601)"].dt.time <= end_time)
]

if df.empty:
    st.warning("No data in this time range.")
    st.stop()

# ---------------------------------------------------------------
# PREPARE TSO DATA
# ---------------------------------------------------------------
# For each TSO we keep two parallel arrays:
#   tso_values[tso]   -> the merged CBMP price (as before)
#   tso_category[tso] -> the direction of that price point:
#                         "up"      -> only the POS (up) price exists     -> aFRR up bid only
#                         "down"    -> only the NEG (down) price exists   -> aFRR down bid only
#                         "neutral" -> both POS and NEG prices exist      -> no net aFRR demand
#                         None      -> neither price exists (no data)
tso_values = {}
tso_category = {}
for tso in TSO_DISPLAY_NAMES.keys():
    neg_col = f"{tso}_NEG"   # down price
    pos_col = f"{tso}_POS"   # up price
    if neg_col in df.columns or pos_col in df.columns:
        vals = []
        cats = []
        for neg, pos in zip(df.get(neg_col, [np.nan]*len(df)), df.get(pos_col, [np.nan]*len(df))):
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
        tso_values[tso] = np.array(vals)
        tso_category[tso] = np.array(cats, dtype=object)

times = df["Zeit (ISO 8601)"]

# ---------------------------------------------------------------
# SELECT TSOs
# ---------------------------------------------------------------
st.subheader("Select TSOs to Display")

available_tsos = [tso for tso in TSO_DISPLAY_NAMES.keys() if tso in tso_values]
selected_tsos = st.multiselect(
    "Choose TSOs",
    options=available_tsos,
    format_func=lambda x: TSO_DISPLAY_NAMES[x],
    default=available_tsos[:6]
)

if not selected_tsos:
    st.warning("Please select at least one TSO.")
    st.stop()

# ---------------------------------------------------------------
# Y-AXIS RANGE
# ---------------------------------------------------------------
all_vals = np.concatenate([tso_values[tso] for tso in selected_tsos])
valid_vals = all_vals[~np.isnan(all_vals)]
if valid_vals.size == 0:
    st.warning("No valid numerical data for the selected TSOs in this time range.")
    st.stop()

data_range_min = float(np.floor(valid_vals.min() - 20))
data_range_max = float(np.ceil(valid_vals.max() + 20))

st.subheader("Y-Axis Range")
user_ymin, user_ymax = st.slider(
    "Select Y-axis range (€/MWh)",
    min_value=data_range_min,
    max_value=data_range_max,
    value=(data_range_min, data_range_max),
    step=0.5
)

# ---------------------------------------------------------------
# MAIN PLOT
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(18, 7))

for tso in selected_tsos:
    ax.plot(times, tso_values[tso], label=TSO_DISPLAY_NAMES[tso], color=TSO_COLORS[tso])

ax.set_title(f"{date_str} Picasso CBMP (Local: {start_time.strftime('%H:%M')} – {end_time.strftime('%H:%M')})")
ax.set_xlabel("Time (Europe/Brussels)")
ax.set_ylabel("€/MWh")
ax.set_ylim(user_ymin, user_ymax)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))
ax.grid(True)
ax.legend(ncols=2)
plt.tight_layout()
st.pyplot(fig)

# ---------------------------------------------------------------
# HELPERS FOR DIRECTION-COLORED (RED/GREEN/BLUE) STAIRCASE PLOTS
# ---------------------------------------------------------------
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

    x = mdates.date2num(times_series.dt.to_pydatetime())
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

# ---------------------------------------------------------------
# INDIVIDUAL SUBPLOTS (colored by aFRR direction)
# ---------------------------------------------------------------
st.header("Individual TSO Graphs")
st.caption("Red = aFRR up bid only · Green = aFRR down bid only · Blue = neutral (both directions present)")

n = len(selected_tsos)
rows = int(np.ceil(n / 2))
fig2, axs = plt.subplots(rows, 2, figsize=(20, 4 * rows), sharex=True, sharey=True)
axs = axs.flatten()

for i, tso in enumerate(selected_tsos):
    ax_i = axs[i]
    plot_direction_staircase(ax_i, times, tso_values[tso], tso_category[tso])
    ax_i.set_title(TSO_DISPLAY_NAMES[tso])
    ax_i.set_ylim(user_ymin, user_ymax)
    ax_i.grid(True)
    ax_i.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))

for j in range(i + 1, len(axs)):
    axs[j].axis("off")

fig2.legend(handles=DIRECTION_LEGEND_ELEMENTS, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
plt.tight_layout()
st.pyplot(fig2)

# ---------------------------------------------------------------
# QUARTER-HOUR AVERAGE PRICES BY DIRECTION (one graph per TSO)
# ---------------------------------------------------------------
st.header("Quarter-Hour Average Prices by Direction")
st.caption(
    "For every 15-minute block, the average price is computed separately for the up, down and "
    "neutral directions (Red = Up, Green = Down, Blue = Neutral). "
    "If a direction has no data in a given quarter-hour, that point is left blank."
)

# Bucket every timestamp into its 15-minute quarter-hour, positionally aligned
# with tso_values / tso_category (same row order as df).
qh_bucket = times.dt.floor("15min").reset_index(drop=True)
qh_index = pd.DatetimeIndex(sorted(qh_bucket.unique()))


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


rows3 = int(np.ceil(n / 2))
fig3, axs3 = plt.subplots(rows3, 2, figsize=(20, 4 * rows3), sharex=True, sharey=True)
axs3 = axs3.flatten()

for i, tso in enumerate(selected_tsos):
    ax_i = axs3[i]
    qh_avgs = qh_direction_averages(tso_values[tso], tso_category[tso], qh_bucket, qh_index)
    ax_i.plot(qh_index, qh_avgs["up"].values, color=DIRECTION_COLORS["up"],
              marker="o", markersize=3, linewidth=1.2, label="Up")
    ax_i.plot(qh_index, qh_avgs["down"].values, color=DIRECTION_COLORS["down"],
              marker="o", markersize=3, linewidth=1.2, label="Down")
    ax_i.plot(qh_index, qh_avgs["neutral"].values, color=DIRECTION_COLORS["neutral"],
              marker="o", markersize=3, linewidth=1.2, label="Neutral")
    ax_i.set_title(TSO_DISPLAY_NAMES[tso])
    ax_i.set_ylim(user_ymin, user_ymax)
    ax_i.grid(True)
    ax_i.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))

for j in range(i + 1, len(axs3)):
    axs3[j].axis("off")

handles3, labels3 = axs3[0].get_legend_handles_labels()
fig3.legend(handles3, labels3, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
plt.tight_layout()
st.pyplot(fig3)

# ---------------------------------------------------------------
# SIMILARITY MATRIX (show ALL values)
# ---------------------------------------------------------------
def percentage_equal(arr1: np.ndarray, arr2: np.ndarray) -> float:
    mask = ~np.isnan(arr1) & ~np.isnan(arr2)
    if mask.sum() == 0:
        return np.nan
    return 100.0 * np.sum(arr1[mask] == arr2[mask]) / mask.sum()

labels = [TSO_DISPLAY_NAMES[t] for t in selected_tsos]
m = len(selected_tsos)
sim_matrix = np.full((m, m), np.nan, dtype=float)

for i in range(m):
    for j in range(m):  # FULL MATRIX NOW
        if i == j:
            sim_matrix[i, j] = 100.0
        else:
            sim_matrix[i, j] = percentage_equal(tso_values[selected_tsos[i]], tso_values[selected_tsos[j]])

sim_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)

st.header("TSO Similarity Matrix (%)")

styled = (
    sim_df.style
    .background_gradient(cmap="RdYlGn", vmin=0, vmax=100, axis=None)
    .format(lambda v: f"{v:.2f}%" if not pd.isna(v) else "")
    .set_properties(**{"text-align": "center"})
    .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
)

st.dataframe(styled, use_container_width=True)
