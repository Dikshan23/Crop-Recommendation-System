import streamlit as st
import sys
import os
from datetime import datetime
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from utils.auth import require_auth, logout_user, init_session, get_user_email
from src.predict import predict_crop

st.set_page_config(page_title="Dashboard", layout="wide")

# --- Initialize session ---
init_session()  # restores user from refresh token

# --- Protect page ---
require_auth()  # only redirect if st.session_state["user"] is None

# --- Sidebar ---
with st.sidebar:
    st.subheader("User Profile")
    user_email = get_user_email(st.session_state.get("user"))
    if user_email:
        st.write(f"👤 {user_email}")
    else:
        st.write("👤 Logged in")  # fallback if email missing

    if st.button("🚪 Logout", use_container_width=True):
        logout_user()
        st.rerun()  # reload page after logout

# --- Page content ---
st.title("Crop Recommendation Dashboard")
st.markdown("Enter the soil and environmental parameters below to get a crop recommendation.")

# --- Prediction form ---
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
        p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0)
        k = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
    with col2:
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    with col3:
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
        rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

    submit = st.form_submit_button("Get Recommendation", use_container_width=True)

if submit:
    with st.spinner("Analyzing soil and weather conditions..."):
        time.sleep(1)
        result = predict_crop(n, p, k, temp, hum, ph, rain)
    st.success("Prediction Result")

    st.info(f"The most suitable crop for these conditions is: {result.upper()}")