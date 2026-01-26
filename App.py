import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Load CSV for selected date and fix time to UTC+1
# ---------------------------------
@st.cache_data(show_spinner=True)
def load_csv_for_date(date_str):
    url = f"https://api.transnetbw.de/picasso-cbmp/csv?date={date_str}&lang=de"
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        st.error("Failed to retrieve data from API.")
        return None
    df = pd.read_csv(io.StringIO(response.content.decode()), sep=';', parse_dates=['Zeit (ISO 8601)'])
    df['Zeit (ISO 8601)'] = df['Zeit (ISO 8601)'] + pd.Timedelta(hours=1)  # Shift to UTC+1
    return df

# ---------------------------------
# 2. Streamlit App
# ---------------------------------
st.title('PICASSO Data Viewer (UTC+1)')

# Sidebar: select date
date_str = st.sidebar.date_input('Select a date', pd.Timestamp.today()).strftime('%Y-%m-%d')

# Load DataFrame
df = load_csv_for_date(date_str)

if df is not None:
    tso_cols = ['ELIA', '50HZT', 'RTE', 'TNL']
    for tso in tso_cols:
        if tso not in df.columns:
            df[tso] = np.nan  # In case column missing
    # Build TSO column data
    tso_values = {tso: df[tso].values for tso in tso_cols}

    # ---------------------------------
    # 3. Sidebar: Time zoom picker
    # ---------------------------------
    st.sidebar.header("Zoom on Time Interval")
    min_time = df['Zeit (ISO 8601)'].min().time()
    max_time = df['Zeit (ISO 8601)'].max().time()
    start_time = st.sidebar.time_input("Start time", value=min_time)
    end_time = st.sidebar.time_input("End time", value=max_time)

    if start_time > end_time:
        st.sidebar.warning("Start time should be before end time.")

    # Filter DataFrame on time interval
    mask = (df['Zeit (ISO 8601)'].dt.time >= start_time) & (df['Zeit (ISO 8601)'].dt.time <= end_time)
    df_zoom = df[mask].copy()

    # Arrays for plotting
    times = df_zoom['Zeit (ISO 8601)']
    elia = df_zoom['ELIA'].values
    hz50 = df_zoom['50HZT'].values
    rte  = df_zoom['RTE'].values
    tnl  = df_zoom['TNL'].values

    # ---------------------------------
    # 4. Plotting
    # ---------------------------------
    st.subheader("Activated mFRR for selected TSOs and time frame")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, elia, label='ELIA')
    ax.plot(times, hz50, label='50HZT')
    ax.plot(times, rte, label='RTE')
    ax.plot(times, tnl, label='TNL')
    ax.set_xlabel("Time")
    ax.set_ylabel("Activated mFRR")
    ax.legend()
    ax.grid()
    fig.autofmt_xdate()
    st.pyplot(fig)

    # ---------------------------------
    # 5. Example: Show raw data table (optional)
    # ---------------------------------
    st.subheader("Filtered Data Table")
    st.dataframe(df_zoom)

else:
    st.warning("No data loaded. Please select a valid date.")
