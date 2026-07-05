import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Set page layout
st.set_page_config(page_title="Nairobi EcoForecast Engine", layout="wide")

st.title("Nairobi Air Quality ARMA Forecast")
st.markdown("Real-time air quality tracking and weather trend forecasting using a custom vector ARMA(1,1) engine.")

# ==========================================
# 1. LOAD PRODUCTION ASSETS VIA JOBLIB
# ==========================================
@st.cache_resource
def load_model_assets():
    model_payload = joblib.load("weights/arma_model2.joblib")
    W = model_payload["weights"]
    b = model_payload["bias"]
    history_df = pd.read_csv("Datafiles/ui_history2.csv")
    return W, b, history_df

try:
    W, b, history_df = load_model_assets()
    st.sidebar.success(" ARMA Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f" Error loading model assets: {e}")
    st.stop()

# ==========================================
# 2. USER INTERACTION INTERFACE
# ==========================================
st.header(" Forecast Controls")

location_map = {
    "Westlands / Parklands (Commercial Hub)": "location_76",
    "Nairobi CBD East (Downtown area)": "location_3966",
    "Nairobi CBD West (Haile Selassie/Moi Avenue Corridor)": "location_3967",
    "Pangani / Muthaiga Border (Urban Transition)": "location_3981"
}

selected_zone = st.selectbox("Select Nairobi Zone:", list(location_map.keys()))
active_loc_column = location_map[selected_zone]


horizon_options = {
    "3 Hours Ahead (Short-term Shock Tracker)": 3,
    "6 Hours Ahead (Mid-term Dynamic Window)": 6,
    "12 Hours Ahead (Standard Half-Day Horizon)": 12,
    "24 Hours Ahead (Full Diurnal Scale / Decay Horizon)": 24
}
selected_horizon_label = st.selectbox("Select Forecast Horizon:", list(horizon_options.keys()), index=2)
forecast_horizon = horizon_options[selected_horizon_label]

# ==========================================
# 3. STREAMING FORECAST ENGINE
# ==========================================
if st.button(" Generate Forecast"):

    loc_flag_cols = ["location_76", "location_3966", "location_3967", "location_3981"]
    loc_mask = history_df[active_loc_column] == 1

    if not loc_mask.any():
        st.error("No baseline data found for this location flag in history.")
        st.stop()

    latest_record = history_df[loc_mask].iloc[-1]

    current_ar_lags = np.array([
        float(latest_record["P0_lag_1h"]),
        float(latest_record["P1_lag_1h"]),
        float(latest_record["P2_lag_1h"]),
        float(latest_record["humidity_lag_1h"]),
        float(latest_record["temperature_lag_1h"])
    ], dtype=np.float64)

    spatial_flags = np.zeros(4, dtype=np.float64)
    spatial_flags[loc_flag_cols.index(active_loc_column)] = 1.0

    last_error = np.zeros(5, dtype=np.float64)
    forecast_results = []

    for hour in range(forecast_horizon):
        full_features = np.hstack((current_ar_lags, spatial_flags, last_error))

        pred_raw = np.dot(full_features, W) + b
        pred_flat = pred_raw.flatten()

        forecast_results.append(pred_flat)

        next_p0 = pred_flat[0]
        next_p1 = pred_flat[1]
        next_p2 = pred_flat[2]
        next_hum_base = current_ar_lags[3] + pred_flat[3]
        next_temp_base = current_ar_lags[4] + pred_flat[4]

        current_ar_lags = np.array([next_p0, next_p1, next_p2, next_hum_base, next_temp_base], dtype=np.float64)
        last_error = np.zeros(5, dtype=np.float64)

    forecast_matrix = np.array(forecast_results)
    columns_y = ["P0", "P1", "P2", "Humidity Change (Δ)", "Temperature Change (Δ)"]
    df_forecast = pd.DataFrame(forecast_matrix, columns=columns_y, index=range(1, forecast_horizon + 1))

    # ==========================================
    # 4. RENDER VISUAL INTERFACE
    # ==========================================
    st.subheader(f" Predictive Tracking: {selected_zone} ({forecast_horizon}-Hour Window)")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Next Hour P0", f"{df_forecast['P0'].iloc[0]:.2f}")
    m2.metric("Next Hour P1", f"{df_forecast['P1'].iloc[0]:.2f}")
    m3.metric("Next Hour P2 (PM2.5)", f"{df_forecast['P2'].iloc[0]:.2f}")
    m4.metric("Humidity Shift", f"{df_forecast['Humidity Change (Δ)'].iloc[0]:+.2f}%")
    m5.metric("Temp Shift", f"{df_forecast['Temperature Change (Δ)'].iloc[0]:+.2f}°C")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Particulate Matter Trends")
        fig_pol, ax_pol = plt.subplots(figsize=(7, 4.5))
        ax_pol.plot(df_forecast.index, df_forecast["P0"], label="P0 (Coarse)", marker='o', color="#2c3e50")
        ax_pol.plot(df_forecast.index, df_forecast["P1"], label="P1 (Medium)", marker='s', color="#27ae60")
        ax_pol.plot(df_forecast.index, df_forecast["P2"], label="P2 (Fine PM2.5)", marker='^', color="#2980b9")
        ax_pol.set_xlabel("Hours Into Future")
        ax_pol.set_ylabel("Concentration Metrics")
        ax_pol.grid(True, alpha=0.3)
        ax_pol.legend()
        st.pyplot(fig_pol)

    with col_right:
        st.markdown("###  Weather Change Velocity Dynamics")
        fig_wth, ax_wth = plt.subplots(figsize=(7, 4.5))
        ax_wth.plot(df_forecast.index, df_forecast["Humidity Change (Δ)"], label="Humidity Δ", marker='o', color="#d35400")
        ax_wth.plot(df_forecast.index, df_forecast["Temperature Change (Δ)"], label="Temperature Δ", marker='x', color="#c0392b")
        ax_wth.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax_wth.set_xlabel("Hours Into Future")
        ax_wth.set_ylabel("Delta Rate per Hour")
        ax_wth.grid(True, alpha=0.3)
        ax_wth.legend()
        st.pyplot(fig_wth)

    st.markdown("###  Raw Multi-Step Prediction Vector Values")
    st.dataframe(df_forecast.style.format("{:.4f}"))

else:
    st.info(" Select your desired neighborhood in the sidebar control panel and click 'Generate Forecast' to trigger the custom matrix execution engine.")