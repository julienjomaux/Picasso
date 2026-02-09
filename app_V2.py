import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date, time
import io
import urllib3
from zoneinfo import ZoneInfo  # ✅ standard library timezone handling (Python 3.9+)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="Picasso CBMP Visualizer", layout="wide")

TSO_COLORS = {
    '50HZT': '#1f77b4',
    'ELIA':  '#ff7f0e',
    'RTE':   '#2ca02c',
    'TNL':   '#d62728'
}

LOCAL_TZ = ZoneInfo("Europe/Brussels")

# --- Top Controls: Date Selection (LOCAL) ---
controls = st.container()
with controls:
    st.header("Settings")
    date_selected = st.date_input(
        label="Select a date (Europe/Brussels)",
        value=datetime.now(LOCAL_TZ).date(),  # ✅ local default date
        min_value=date(2020, 1, 1)
    )
    date_str = date_selected.strftime("%Y-%m-%d")

st.title(f"Picasso CBMP Data for {date_str} (local time)")

# --- Data Download/Load ---
@st.cache_data(ttl=300, show_spinner=True)  # expires after 5 minutes
def load_csv_for_date(date_str: str) -> pd.DataFrame | None:
    """
    Loads the CSV for a given date from the TransnetBW Picasso CBMP API
    and converts the timestamp to Europe/Brussels local time, properly
    handling daylight saving time changes.
    """
    url = f"https://api.transnetbw.de/picasso-cbmp/csv?date={date_str}&lang=de"
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        st.error(f"Failed to retrieve data from API. Status code: {response.status_code}")
        return None

    # Parse CSV; we will re-parse the time column to ensure tz-aware UTC, then convert to local
    df = pd.read_csv(io.StringIO(response.content.decode()), sep=';', parse_dates=['Zeit (ISO 8601)'])

    # ✅ Ensure timestamps are treated as UTC, then convert to Europe/Brussels
    # This automatically applies +1h in winter and +2h in summer.
    dtcol = 'Zeit (ISO 8601)'
    df[dtcol] = pd.to_datetime(df[dtcol], utc=True).dt.tz_convert(LOCAL_TZ)

    return df

df_raw = load_csv_for_date(date_str)

if df_raw is not None and not df_raw.empty:
    # --- Determine available local time range for the selected date ---
    # Note: The API returns the selected UTC date; after conversion to local time,
    # the local times may slightly spill over to the previous/next local day at the edges.
    # We set the slider bounds to the actual min/max local times present in the data.
    local_times = df_raw['Zeit (ISO 8601)'].dt.time
    min_local_time = min(local_times)
    max_local_time = max(local_times)

    # --- Top Controls: Time Range (LOCAL, adapted to DST day length) ---
    with controls:
        st.subheader("Time Range (local)")
        time_range = st.slider(
            "Select Hour Range",
            min_value=min_local_time,
            max_value=max_local_time,
            value=(min_local_time, max_local_time),
            format="HH:mm"
        )
    start_time, end_time = time_range

    # --- Filter by local time range ---
    df = df_raw[
        (df_raw['Zeit (ISO 8601)'].dt.time >= start_time) &
        (df_raw['Zeit (ISO 8601)'].dt.time <= end_time)
    ].copy()

    if df.empty:
        st.warning("No data found for the selected time range.")
    else:
        # --- Prepare TSO Data ---
        tso_names = sorted(set(col.split('_')[0] for col in df.columns if '_' in col))
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

        times = df['Zeit (ISO 8601)']  # tz-aware local datetimes

        # --- Top Controls: TSO + Line Style (Dashed) Selection ---
        with controls:
            st.subheader("TSOs to Display")
            cols = st.columns(4)
            tso_settings = {}
            for i, tso in enumerate(TSO_COLORS.keys()):
                with cols[i]:
                    show = st.checkbox(f"{tso}", value=True, key=f"show_{tso}")
                    dash = st.checkbox("Dashed", value=False, key=f"dash_{tso}")
                    tso_settings[tso] = {'show': show, 'dash': dash}

        # --- Compute y-axis range based on selected TSOs ---
        all_vals = []
        for tso, settings in tso_settings.items():
            if settings['show'] and tso in tso_values:
                all_vals.append(tso_values[tso])

        if not all_vals:
            st.warning("No TSOs selected/displayed.")
            st.stop()

        all_vals = np.concatenate(all_vals)
        valid = ~np.isnan(all_vals)
        if not np.any(valid):
            st.warning("No valid data available for selected TSOs in this time range.")
            st.stop()

        y_min_data = float(np.nanmin(all_vals))
        y_max_data = float(np.nanmax(all_vals))
        y_pad = (y_max_data - y_min_data) * 0.05 if y_max_data > y_min_data else 1.0

        # --- Top Controls: Y-axis Range Controls (default = min-20, max+20) ---
        with controls:
            st.subheader("Y-Axis Limits")
            data_range_min = float(np.floor(y_min_data - 20))
            data_range_max = float(np.ceil(y_max_data + 20))
            user_ymin, user_ymax = st.slider(
                "Select Y-Axis Range (€/MWh)",
                min_value=data_range_min,
                max_value=data_range_max,
                value=(data_range_min, data_range_max),
                step=0.5
            )

        # --- Main Plot ---
        fig, ax = plt.subplots(figsize=(18, 7))
        for tso, settings in tso_settings.items():
            if settings['show'] and tso in tso_values:
                linestyle = '--' if settings['dash'] else '-'
                ax.plot(times, tso_values[tso], label=tso, color=TSO_COLORS[tso], linestyle=linestyle)

        ax.legend()
        ax.set_xlabel("Time (Europe/Brussels)")
        ax.set_ylabel("€/MWh")
        ax.set_title(
            f"{date_str} Picasso CBMP "
            f"(Local: {start_time.strftime('%H:%M')} – {end_time.strftime('%H:%M')})"
        )
        # ✅ Use local timezone-aware formatter
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=LOCAL_TZ))
        ax.set_ylim(user_ymin, user_ymax)
        ax.grid(True, which='major', axis='both')
        plt.tight_layout()
        st.pyplot(fig)

        # --- 2x2 Subplots (forcerange) ---
        fig_sub, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
        axs = axs.flat
        tsos_order = ['50HZT', 'ELIA', 'RTE', 'TNL']
        for i, tso in enumerate(tsos_order):
            ax_subplot = axs[i]
            if tso in tso_values and tso_settings[tso]['show']:
                linestyle = '--' if tso_settings[tso]['dash'] else '-'
                ax_subplot.plot(times, tso_values[tso], color=TSO_COLORS[tso], linestyle=linestyle, label=tso)
                ax_subplot.set_title(tso)
                ax_subplot.set_ylabel("€/MWh")
                ax_subplot.legend()
                ax_subplot.set_ylim(user_ymin, user_ymax)
            else:
                ax_subplot.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax_subplot.set_ylim(user_ymin, user_ymax)

            ax_subplot.grid(True, which='major')
            ax_subplot.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax_subplot.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=LOCAL_TZ))

        for ax_subplot in axs[2:]:
            ax_subplot.set_xlabel("Time (Europe/Brussels)")

        plt.tight_layout()
        st.pyplot(fig_sub)

        # --- Statistics ---
        def percentage_equal(arr1, arr2):
            mask = ~np.isnan(arr1) & ~np.isnan(arr2)
            if mask.sum() == 0:
                return 0.0
            return 100 * (arr1[mask] == arr2[mask]).sum() / mask.sum()

        elia = tso_values.get('ELIA', np.full(len(times), np.nan))
        hz50 = tso_values.get('50HZT', np.full(len(times), np.nan))
        rte  = tso_values.get('RTE', np.full(len(times), np.nan))
        tnl  = tso_values.get('TNL', np.full(len(times), np.nan))

        mask_unique = (
            ~np.isnan(elia) & ~np.isnan(hz50) & ~np.isnan(rte) & ~np.isnan(tnl) &
            (elia != hz50) & (elia != rte) & (elia != tnl)
        )
        denom = np.sum(~np.isnan(elia) & ~np.isnan(hz50) & ~np.isnan(rte) & ~np.isnan(tnl))
        percent_elia_unique = 100 * np.sum(mask_unique) / denom if denom > 0 else 0

        st.markdown("### Key Results (for selected range)")
        st.write(f"**Percentage of time ELIA is unique:** {percent_elia_unique:.2f}%")
        st.write(f"**Percentage of time ELIA = 50 Hertz:** {percentage_equal(elia, hz50):.2f}%")
        st.write(f"**Percentage of time ELIA = RTE:** {percentage_equal(elia, rte):.2f}%")
        st.write(f"**Percentage of time ELIA = Tennet NL:** {percentage_equal(elia, tnl):.2f}%")

else:
    st.warning("No data available for the selected date.")