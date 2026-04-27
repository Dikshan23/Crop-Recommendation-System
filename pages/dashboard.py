import streamlit as st
import sys
import os
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from utils.auth import require_auth, logout_user, init_session, get_user_email
from src.predict import predict_crop_with_confidence
from src.validations import validate_inputs, warn_inputs
from src.history_predict import save_prediction_to_history, get_user_prediction_history, get_prediction_count, delete_prediction

st.set_page_config(page_title="Dashboard", layout="wide")

# Input boundaries
BOUNDARIES = {
    'N': (0, 140),
    'P': (5, 145),
    'K': (5, 205),
    'temperature': (8, 45),
    'humidity': (10, 100),
    'ph': (3, 10),
    'rainfall': (20, 300)
}

CONFIDENCE_THRESHOLD = 0.80

# Validation function
def validate_form(n, p, k, temp, hum, ph, rain):
    """Validate inputs using the shared backend ranges and messages."""
    return validate_inputs(n, p, k, temp, hum, ph, rain)

# Initialize session
init_session()
require_auth()

# Sidebar
with st.sidebar:
    st.subheader("User Profile")
    user_email = get_user_email(st.session_state.get("user"))
    st.write(f"👤 {user_email}" if user_email else "👤 Logged in")

    if st.button("Logout", use_container_width=True):
        logout_user()
        st.rerun()

# Page content
st.title("Crop Recommendation Dashboard")

user = st.session_state.get("user")

# Session state for prediction count
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = get_prediction_count(user)

# Session state for input tracking
if "prev_inputs" not in st.session_state:
    st.session_state.prev_inputs = None

# Create tabs
tab1, tab2 = st.tabs(["Make Prediction", f"History ({st.session_state.prediction_count})"])

# Tab 1: Prediction form
with tab1:
    st.markdown("Enter the soil and environmental parameters below to get a crop recommendation.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            n = st.number_input("Nitrogen (N)", value=50.0)
            p = st.number_input("Phosphorus (P)", value=50.0)
            k = st.number_input("Potassium (K)", value=50.0)
        with col2:
            temp = st.number_input("Temperature (°C)", value=25.0)
            hum = st.number_input("Humidity (%)", value=60.0)
        with col3:
            ph = st.number_input("Soil pH", value=6.5)
            rain = st.number_input("Rainfall (mm)", value=100.0)

        # Detect if user changed any input → clear stale result
        current_inputs = (n, p, k, temp, hum, ph, rain)
        if current_inputs != st.session_state.prev_inputs:
            st.session_state.prev_inputs = current_inputs

        submit = st.form_submit_button("Get Recommendation", use_container_width=True)

    # Run when submit is clicked
    if submit:
        with st.spinner("Analyzing soil and weather conditions..."):
            time.sleep(1)
            result = predict_crop(n, p, k, temp, hum, ph, rain)
        
        st.success("Prediction Result")
        st.info(f"The most suitable crop for these conditions is: {result.upper()}")
        
        # Save to history
        if save_prediction_to_history(user, n, p, k, temp, hum, ph, rain, result):
            st.success("✅ Prediction saved to history!")
            st.balloons()
        else:
            st.warning("⚠️ Prediction made but could not save to history")

            # Low confidence result
            else:
                st.error("🚫 Sorry, we cannot recommend a crop for these conditions.")

# Tab 2: Prediction history
with tab2:
    history = get_user_prediction_history(user)

    # Sync count with actual DB
    if history is not None:
        st.session_state.prediction_count = len(history)

    if history:
        df_history = pd.DataFrame(history)

        # Top stats
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("📊 Total", len(history))
        with stat_col2:
            st.metric("🌾 Crops", df_history["predicted_crop"].nunique())
        with stat_col3:
            st.metric("🏆 Top", df_history["predicted_crop"].mode()[0].title())

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
            for pred in filtered:
                date_obj = pd.to_datetime(pred["created_at"])
                nepal_tz = pytz.timezone('Asia/Kathmandu')
                date_obj = date_obj.replace(tzinfo=pytz.UTC).astimezone(nepal_tz)
                date_str = date_obj.strftime("%b %d, %Y")
                time_str = date_obj.strftime("%H:%M")
                crop = pred["predicted_crop"].title()

                # Confidence in title
                conf_str = ""
                if pred.get("confidence") is not None and pred["confidence"] >= 0:
                    conf_val = pred["confidence"] * 100 if pred["confidence"] <= 1 else pred["confidence"]
                    conf_str = f" ({conf_val:.0f}%)"

                with st.expander(f"🌾 {crop}{conf_str} — {date_str} at {time_str}", expanded=False):
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

                        if pred.get("confidence") is not None and pred["confidence"] >= 0:
                            conf_val = pred["confidence"] * 100 if pred["confidence"] <= 1 else pred["confidence"]
                            st.write(f"🎯 **Confidence:** {conf_val:.1f}%")

                    # Delete button
                    st.markdown("---")
                    col_delete, col_space = st.columns([1, 4])
                    with col_delete:
                        if st.button("🗑️ Delete", key=f"delete_{pred['id']}", use_container_width=True):
                            if delete_prediction(pred['id']):
                                st.session_state.prediction_count -= 1
                                st.success("Prediction deleted!")
                                time.sleep(0.3)
                                st.rerun()
                            else:
                                st.error("Failed to delete prediction")
        else:
            st.warning("No predictions found")
    else:
        st.info("📭 No prediction history yet. Make a prediction to get started!")
