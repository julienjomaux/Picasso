import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import io

st.set_page_config(page_title="Picasso CBMP Visualizer", layout="wide")

# --- Streamlit Sidebar: Date selection ---
st.sidebar.header("Date Selection")
date_str = st.sidebar.date_input(
    label="Select a date",
    value=datetime.utcnow(),
    min_value=datetime(2020, 1, 1)  # Adapt if earlier is possible
).strftime("%Y-%m-%d")

st.title(f"Picasso CBMP Data for {date_str}")

# --- Download / Load Data ---
@st.cache_data(show_spinner=True)
def load_csv_for_date(date_str):
    url = f"https://api.transnetbw.de/picasso-cbmp/csv?date={date_str}&lang=de"
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        st.error("Failed to retrieve data from API.")
        return None
    df = pd.read_csv(io.StringIO(response.content.decode()), sep=';', parse_dates=['Zeit (ISO 8601)'])
    df['Zeit (ISO 8601)'] = df['Zeit (ISO 8601)'] + pd.Timedelta(hours=1)

    return df

df = load_csv_for_date(date_str)

if df is not None:
    # --- TSO Names ---
    tso_names = sorted(set(col.split('_')[0] for col in df.columns if '_' in col))
    
    # --- Prepare dictionary ---
    tso_values = {}
    for tso in tso_names:
        neg_col = f"{tso}_NEG"
        pos_col = f"{tso}_POS"
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

    # --- Extract and Plot ---
    times = df['Zeit (ISO 8601)']
    to_plot = ['50HZT', 'ELIA', 'RTE', 'TNL']
    elia = tso_values.get('ELIA', np.full(len(times), np.nan))
    hz50 = tso_values.get('50HZT', np.full(len(times), np.nan))
    rte  = tso_values.get('RTE', np.full(len(times), np.nan))
    tnl  = tso_values.get('TNL', np.full(len(times), np.nan))
    mask_unique = (
        ~np.isnan(elia) & ~np.isnan(hz50) & ~np.isnan(rte) & ~np.isnan(tnl) &
        (elia != hz50) & (elia != rte) & (elia != tnl)
    )
    
    fig, ax = plt.subplots(figsize=(18, 7))
    for tso in to_plot:
        ax.plot(times, tso_values[tso], label=tso)
    ax.legend()
    ax.set_xlabel("")
    ax.set_ylabel("€/MWh")
    ax.set_title(f"{date_str} Picasso CBMP")
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, which='major', axis='both')
    plt.tight_layout()
    st.pyplot(fig)

    # --- Statistics ---
    def percentage_equal(arr1, arr2):
        mask = ~np.isnan(arr1) & ~np.isnan(arr2)
        if mask.sum() == 0:
            return 0.0
        return 100 * (arr1[mask] == arr2[mask]).sum() / mask.sum()
    
    perc_elia_50HZT = percentage_equal(elia, hz50)
    perc_elia_RTE = percentage_equal(elia, rte)
    perc_elia_TNL = percentage_equal(elia, tnl)
    percent_elia_unique = 100 * np.sum(mask_unique) / np.sum(~np.isnan(elia) & ~np.isnan(hz50) & ~np.isnan(rte) & ~np.isnan(tnl))

    st.markdown("### Key Results")
    st.write(f"**Percentage of time ELIA is unique:** {percent_elia_unique:.2f}%")
    st.write(f"**Percentage of time ELIA = 50HZT:** {perc_elia_50HZT:.2f}%")
    st.write(f"**Percentage of time ELIA = RTE:** {perc_elia_RTE:.2f}%")
    st.write(f"**Percentage of time ELIA = TNL:** {perc_elia_TNL:.2f}%")
else:
    st.warning("No data available for the selected date.")
