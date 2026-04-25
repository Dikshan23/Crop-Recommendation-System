import streamlit as st
import sys
import os
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.auth import require_auth, logout_user, init_session, get_user_email
from src.predict import predict_crop
from src.history_predict import save_prediction_to_history, get_user_prediction_history, get_prediction_count, delete_prediction
from src.validations import validate_inputs, warn_inputs, VALID_RANGES

st.set_page_config(page_title="Dashboard", layout="wide")

init_session()
require_auth()

# --- Sidebar ---
with st.sidebar:
    st.subheader("User Profile")
    user_email = get_user_email(st.session_state.get("user"))
    st.write(f"👤 {user_email}" if user_email else "👤 Logged in")

    if st.button("Logout", use_container_width=True):
        logout_user()
        st.rerun()

st.title("Crop Recommendation Dashboard")

user = st.session_state.get("user")
prediction_count = get_prediction_count(user)

tab1, tab2 = st.tabs(["Make Prediction", f"History ({prediction_count})"])

# ─────────────────────────────────────────────
# TAB 1: PREDICTION FORM
# ─────────────────────────────────────────────
with tab1:
    st.markdown("Enter the soil and environmental parameters below to get a crop recommendation.")

    def range_hint(key):
        r = VALID_RANGES[key]
        unit = f" {r['unit']}" if r["unit"] else ""
        return f"Valid range: {r['min']} – {r['max']}{unit}"

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            n = st.number_input("Nitrogen (N)",    min_value=0.0,  max_value=200.0, value=50.0,  help=range_hint("nitrogen"))
            p = st.number_input("Phosphorus (P)",  min_value=0.0,  max_value=200.0, value=50.0,  help=range_hint("phosphorus"))
            k = st.number_input("Potassium (K)",   min_value=0.0,  max_value=200.0, value=50.0,  help=range_hint("potassium"))

        with col2:
            temp = st.number_input("Temperature (°C)", min_value=0.0,  max_value=60.0,  value=25.0,  help=range_hint("temperature"))
            hum  = st.number_input("Humidity (%)",     min_value=0.0,  max_value=100.0, value=60.0,  help=range_hint("humidity"))

        with col3:
            ph   = st.number_input("Soil pH",          min_value=0.0,  max_value=14.0,  value=6.5,   help=range_hint("ph"))
            rain = st.number_input("Rainfall (mm)",    min_value=0.0,  max_value=500.0, value=100.0, help=range_hint("rainfall"))

        submit = st.form_submit_button("Get Recommendation", use_container_width=True)

    if submit:
        # --- Hard validation ---
        errors = validate_inputs(n, p, k, temp, hum, ph, rain)

        if errors:
            st.error("Please fix the following input errors before proceeding:")
            for err in errors:
                st.warning(f"⚠️ {err}")

        else:
            # --- Soft warnings (borderline / unusual values) ---
            soft_warnings = warn_inputs(n, p, k, temp, hum, ph, rain)
            if soft_warnings:
                with st.expander("⚠️ Input Warnings — values accepted but may affect confidence", expanded=True):
                    for w in soft_warnings:
                        st.warning(w)

            # --- Prediction ---
            try:
                with st.spinner("Analyzing soil and weather conditions..."):
                    time.sleep(1)
                    crop, confidence, proba = predict_crop(n, p, k, temp, hum, ph, rain)

                st.success("Prediction Result")

                res_col1, res_col2 = st.columns([1.2, 1])

                with res_col1:
                    st.markdown(f"### 🌾 Recommended Crop: **{crop.upper()}**")

                with res_col2:
                    pct = round(confidence * 100, 1)

                    # Confidence label and color
                    if pct >= 80:
                        conf_label = "High Confidence"
                        conf_color = "green"
                    elif pct >= 55:
                        conf_label = "Moderate Confidence"
                        conf_color = "orange"
                    else:
                        conf_label = "Low Confidence"
                        conf_color = "red"

                    st.markdown(f"### 📊 Confidence: `{pct}%` — :{conf_color}[{conf_label}]")
                    st.progress(confidence)

                    if pct < 55:
                        st.warning(
                            "⚠️ Low confidence prediction. The input values may not closely match "
                            "any crop's ideal conditions. Consider consulting an agronomist."
                        )

                # Show top alternative crops if confidence < 80
                if confidence < 0.80 and proba:
                    sorted_proba = sorted(proba.items(), key=lambda x: x[1], reverse=True)
                    top_alternatives = [(c, p) for c, p in sorted_proba if c != str(crop)][:3]

                    if top_alternatives:
                        with st.expander("🔍 Other possible crops (ranked by probability)", expanded=False):
                            for alt_crop, alt_prob in top_alternatives:
                                alt_pct = round(alt_prob * 100, 1)
                                st.write(f"🌱 **{alt_crop.title()}** — {alt_pct}%")
                                st.progress(alt_prob)

                # Save to history
                if save_prediction_to_history(user, n, p, k, temp, hum, ph, rain, crop):
                    st.success("✅ Prediction saved to history!")
                else:
                    st.warning("⚠️ Prediction made but could not save to history")

            except ValueError as e:
                st.error(f"Validation error: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ─────────────────────────────────────────────
# TAB 2: PREDICTION HISTORY
# ─────────────────────────────────────────────
with tab2:
    history = get_user_prediction_history(user)

    if history:
        df_history = pd.DataFrame(history)

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("📊 Total", len(history))
        with stat_col2:
            st.metric("🌾 Crops", df_history["predicted_crop"].nunique())
        with stat_col3:
            most_common = df_history["predicted_crop"].mode()[0].title()
            st.metric("🏆 Top", most_common)

        st.markdown("")

        col1, col2 = st.columns([2, 1])
        with col1:
            search = st.text_input("🔍 Search crop...", placeholder="Rice, Wheat, etc")
        with col2:
            sort_by = st.selectbox("Sort", ["Newest", "Oldest"])

        filtered = history.copy()
        if search:
            filtered = [p for p in filtered if search.lower() in p["predicted_crop"].lower()]

        filtered = sorted(
            filtered,
            key=lambda x: pd.to_datetime(x["created_at"]),
            reverse=(sort_by == "Newest")
        )

        if filtered:
            for pred in filtered:
                date_obj = pd.to_datetime(pred["created_at"])
                date_str = date_obj.strftime("%b %d, %Y")
                time_str = date_obj.strftime("%H:%M")
                crop     = pred["predicted_crop"].title()

                with st.expander(f"🌾 {crop} — {date_str} at {time_str}", expanded=False):
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

                    st.markdown("---")
                    col_delete, _ = st.columns([1, 4])
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