import streamlit as st
import sys
import os
from datetime import datetime
import time
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from utils.auth import require_auth, logout_user, init_session, get_user_email
from src.predict import predict_crop
from src.history_predict import save_prediction_to_history, get_user_prediction_history, get_prediction_count, delete_prediction

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

    if st.button("Logout", use_container_width=True):
        logout_user()
        st.rerun()  # reload page after logout

# --- Page content ---
st.title("Crop Recommendation Dashboard")

# Get user for history tracking
user = st.session_state.get("user")
prediction_count = get_prediction_count(user)

# Create tabs
tab1, tab2 = st.tabs(["Make Prediction", f"History ({prediction_count})"])

# --- TAB 1: PREDICTION FORM ---
with tab1:
    st.markdown("Enter the soil and environmental parameters below to get a crop recommendation.")
    
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
        
        # Save to history
        if save_prediction_to_history(user, n, p, k, temp, hum, ph, rain, result):
            st.success("✅ Prediction saved to history!")
        else:
            st.warning("⚠️ Prediction made but could not save to history")

# --- TAB 2: PREDICTION HISTORY ---
with tab2:
    # Fetch user's prediction history
    history = get_user_prediction_history(user)
    
    if history:
        df_history = pd.DataFrame(history)
        
        # Top stats - simple
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("📊 Total", len(history))
        with stat_col2:
            crops = df_history["predicted_crop"].nunique()
            st.metric("🌾 Crops", crops)
        with stat_col3:
            most_common = df_history["predicted_crop"].mode()[0].title()
            st.metric("🏆 Top", most_common)
        
        st.markdown("")
        
        # Search and sort
        col1, col2 = st.columns([2, 1])
        with col1:
            search = st.text_input("🔍 Search crop...", placeholder="Rice, Wheat, etc")
        with col2:
            sort_by = st.selectbox("Sort", ["Newest", "Oldest"])
        
        # Filter and sort
        filtered = history.copy()
        if search:
            filtered = [p for p in filtered if search.lower() in p["predicted_crop"].lower()]
        
        filtered = sorted(filtered, key=lambda x: pd.to_datetime(x["created_at"]), reverse=(sort_by == "Newest"))
        
        if filtered:
            for idx, pred in enumerate(filtered):
                date_obj = pd.to_datetime(pred["created_at"])
                date_str = date_obj.strftime("%b %d, %Y")
                time_str = date_obj.strftime("%H:%M")
                crop = pred["predicted_crop"].title()
                
                # Create expandable section with simple title
                with st.expander(f"🌾 {crop} — {date_str} at {time_str}", expanded=False):
                    # Show full details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Soil Nutrients")
                        st.write(f"🧪 **Nitrogen (N):** {pred['nitrogen']:.1f}")
                        st.write(f"🧪 **Phosphorus (P):** {pred['phosphorus']:.1f}")
                        st.write(f"🧪 **Potassium (K):** {pred['potassium']:.1f}")
                        
                        st.markdown("### Soil Chemistry")
                        st.write(f"🧫 **pH Level:** {pred['ph']:.2f}")
                    
                    with col2:
                        st.markdown("### Environment")
                        st.write(f"🌡️ **Temperature:** {pred['temperature']:.1f}°C")
                        st.write(f"💧 **Humidity:** {pred['humidity']:.1f}%")
                        st.write(f"🌧️ **Rainfall:** {pred['rainfall']:.1f}mm")
                        
                        st.markdown("### Summary")
                        npk_total = pred['nitrogen'] + pred['phosphorus'] + pred['potassium']
                        st.write(f"📊 **Total NPK:** {npk_total:.1f}")
                    
                    # Delete button
                    st.markdown("---")
                    col_delete, col_space = st.columns([1, 4])
                    with col_delete:
                        if st.button("🗑️ Delete", key=f"delete_{pred['id']}", use_container_width=True):
                            if delete_prediction(pred['id']):
                                st.success("Prediction deleted!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to delete prediction")
        else:
            st.warning("No predictions found")
    else:
        st.info("📭 No prediction history yet. Make a prediction to get started!")
