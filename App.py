import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time
import io

st.set_page_config(page_title="Picasso CBMP Visualizer", layout="wide")

# --- Fixed Configuration ---
# ADDED: Mapping to ensure each TSO always has the same color
TSO_COLORS = {
    '50HZT': '#1f77b4',  # Blue
    'ELIA':  '#ff7f0e',  # Orange
    'RTE':   '#2ca02c',  # Green
    'TNL':   '#d62728'   # Red
}
DEFAULT_TSOS = list(TSO_COLORS.keys())

# --- Streamlit Sidebar: Selection ---
st.sidebar.header("Settings")
date_str = st.sidebar.date_input(
    label="Select a date",
    value=datetime.utcnow(),
    min_value=datetime(2020, 1, 1)
).strftime("%Y-%m-%d")

time_range = st.sidebar.slider(
    "Select Hour Range",
    value=(time(0, 0), time(23, 59)),
    format="HH:mm"
)

# ADDED: Individual checkboxes for each TSO
st.sidebar.subheader("TSO Selection")
selected_tsos = []
for tso in DEFAULT_TSOS:
    if st.sidebar.checkbox(tso, value=True):
        selected_tsos.append(tso)

st.title(f"Picasso CBMP Data for {date_str}")

# --- Download / Load Data ---
@st.cache_data(show_spinner=True)
def load_csv_for_date(date_str):
    url = f"https://api.transnetbw.de{date_str}&lang=de"
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        st.error("Failed to retrieve data from API.")
        return None
    df = pd.read_csv(io.StringIO(response.content.decode()), sep=';', parse_dates=['Zeit (ISO 8601)'])
    df['Zeit (ISO 8601)'] = df['Zeit (ISO 8601)'] + pd.Timedelta(hours=1)
    return df

df_raw = load_csv_for_date(date_str)

if df_raw is not None:
    start_time, end_time = time_range
    df = df_raw[
        (df_raw['Zeit (ISO 8601)'].dt.time >= start_time) & 
        (df_raw['Zeit (ISO 8601)'].dt.time <= end_time)
    ].copy()

    if df.empty:
        st.warning("No data found for the selected time range.")
    elif not selected_tsos:
        st.warning("Please tick at least one TSO in the sidebar.")
    else:
        # --- Prepare dictionary ---
        tso_values = {}
        for tso in selected_tsos:
            neg_col, pos_col = f"{tso}_NEG", f"{tso}_POS"
            if neg_col in df.columns or pos_col in df.columns:
                vals = []
                for neg, pos in zip(df.get(neg_col, [np.nan]*len(df)), df.get(pos_col, [np.nan]*len(df))):
                    neg_val = np.nan if pd.isna(neg) or neg == 'N/A' else float(neg)
                    pos_val = np.nan if pd.isna(pos) or pos == 'N/A' else float(pos)
                    if np.isnan(neg_val) and np.isnan(pos_val):
                        vals.append(np.nan)
                    elif np.isnan(neg_val):
                        vals.append(pos_val)
                    elif np.isnan(pos_val):
                        vals.append(neg_val)
                    else:
                        vals.append((neg_val + pos_val) / 2)
                tso_values[tso] = np.array(vals)

        # --- Plotting ---
        times = df['Zeit (ISO 8601)']
        fig, ax = plt.subplots(figsize=(18, 7))
        
        for tso in selected_tsos:
            if tso in tso_values:
                # ADDED: Uses color from fixed TSO_COLORS mapping
                ax.plot(times, tso_values[tso], label=tso, color=TSO_COLORS[tso])
        
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("€/MWh")
        ax.set_title(f"{date_str} Picasso CBMP")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

        # --- Statistics ---
        def percentage_equal(arr1, arr2):
            mask = ~np.isnan(arr1) & ~np.isnan(arr2)
            return 100 * (arr1[mask] == arr2[mask]).sum() / mask.sum() if mask.sum() > 0 else 0.0

        st.markdown("### Statistics (Current Selection)")
        if 'ELIA' in tso_values:
            elia = tso_values['ELIA']
            for other in [t for t in selected_tsos if t != 'ELIA']:
                st.write(f"**ELIA = {other}:** {percentage_equal(elia, tso_values[other]):.2f}%")
        else:
            st.info("Tick 'ELIA' to see comparison statistics.")
else:
    st.warning("No data available for the selected date.")
