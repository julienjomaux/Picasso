import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date
import io
import urllib3
from zoneinfo import ZoneInfo
import itertools

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="Picasso CBMP Visualizer", layout="wide")

LOCAL_TZ = ZoneInfo("Europe/Brussels")

# ---------------------------------------------------------------
# TSO definitions (updated: added AMP, TNG, TTG; CEPX -> CEPS)
# ---------------------------------------------------------------
TSO_DISPLAY_NAMES = {
    "50HZT":  "50HZT (Germany)",
    "APG":    "APG (Austria)",
    "ELIA":   "Elia (Belgium)",
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

# ---------------------------------------------------------------
# SETTINGS UI (top of page)
# ---------------------------------------------------------------
st.header("Settings")
date_selected = st.date_input(
    "Select a date (Europe/Brussels)",
    value=datetime.now(LOCAL_TZ).date(),
    min_value=date(2020, 1, 1)
)
date_str = date_selected.strftime("%Y-%m-%d")
st.title(f"Picasso CBMP Data for {date_str} (local time)")

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
    # Ensure UTC then convert to local tz
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
# PREPARE TSO DATA (mean of NEG/POS, ignoring N/A and NaN)
# ---------------------------------------------------------------
tso_values = {}
for tso in TSO_DISPLAY_NAMES.keys():
    neg_col = f"{tso}_NEG"
    pos_col = f"{tso}_POS"
    if neg_col in df.columns or pos_col in df.columns:
        vals = []
        for neg, pos in zip(df.get(neg_col, [np.nan]*len(df)), df.get(pos_col, [np.nan]*len(df))):
            neg_val = np.nan if pd.isna(neg) or neg == "N/A" else float(neg)
            pos_val = np.nan if pd.isna(pos) or pos == "N/A" else float(pos)
            if np.isnan(neg_val) and np.isnan(pos_val):
                vals.append(np.nan)
            elif np.isnan(neg_val):
                vals.append(pos_val)
            elif np.isnan(pos_val):
                vals.append(neg_val)
            else:
                vals.append((neg_val + pos_val) / 2)
        tso_values[tso] = np.array(vals)

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
    default=available_tsos[:6]  # default to first six available
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
# MAIN PLOT (all selected TSOs)
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
# INDIVIDUAL SUBPLOTS (two per row)
# ---------------------------------------------------------------
st.header("Individual TSO Graphs")

n = len(selected_tsos)
rows = int(np.ceil(n / 2))
fig2, axs = plt.subplots(rows, 2, figsize=(20, 4 * rows), sharex=True, sharey=True)
axs = axs.flatten()

for i, tso in enumerate(selected_tsos):
    ax_i = axs[i]
    ax_i.plot(times, tso_values[tso], color=TSO_COLORS[tso])
    ax_i.set_title(TSO_DISPLAY_NAMES[tso])
    ax_i.set_ylim(user_ymin, user_ymax)
    ax_i.grid(True)
    ax_i.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))

# Hide any unused axes if odd number
for j in range(i + 1, len(axs)):
    axs[j].axis("off")

plt.tight_layout()
st.pyplot(fig2)

# ---------------------------------------------------------------
# SIMILARITY MATRIX (percentage of equality pairwise)
# - Compare only among selected TSOs
# - Diagonal = 100%
# - Upper triangle hidden (NaN -> blank in Styler)
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
    for j in range(i + 1):  # fill diagonal and lower triangle only
        if i == j:
            sim_matrix[i, j] = 100.0
        else:
            sim_matrix[i, j] = percentage_equal(tso_values[selected_tsos[i]], tso_values[selected_tsos[j]])
        # Upper triangle (j > i) remains NaN by design

sim_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)

st.header("TSO Similarity Matrix (%)")
styled = (
    sim_df.style
    .format(lambda v: "" if pd.isna(v) else f"{v:.2f}%")
    .set_properties(**{"text-align": "center"})
    .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
)
st.dataframe(styled, use_container_width=True)
