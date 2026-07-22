import streamlit as st
import sys
import os
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.auth import require_auth, confirm_logout_dialog, init_session, get_user_email, get_user_fullname, is_admin
from src.predict import predict_crop          # only predict_crop, no adjusted_confidence
from src.validations import warn_inputs
from src.history_predict import (
    save_prediction_to_history,
    get_user_prediction_history,
    get_prediction_count,
    delete_prediction
)

st.set_page_config(page_title="Dashboard", layout="wide")


st.markdown("""
<style>
.st-key-logout_btn_wrapper button {
    background: green;
    color: white;
    border: 1px solid #ced4da;
}
.st-key-logout_btn_wrapper button:hover {
    background: darkgreen;
    border-color: #adb5bd;
    color: #f1f1f1;
}
</style>
""", unsafe_allow_html=True)

# --- Initialize session ---
init_session()

# --- Protect page ---
require_auth()

if is_admin():
    st.switch_page("pages/admin_home.py")

# --- Get user info ---
user = st.session_state.get("user")
user_fullname = get_user_fullname(user)
user_email = get_user_email(user)

# --- Top header with welcome message and logout button ---
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.title(f"Welcome, {user_fullname}!")

with col3:
    with st.container(key="logout_btn_wrapper"):
        if st.button("Logout", width = "stretch", key="cancel_logout_btn"):
            confirm_logout_dialog()

if is_admin():
    st.page_link("pages/admin_home.py", label="Admin Panel", icon=":material/admin_panel_settings:")
    st.caption("Admin access enabled")
        

st.divider()

# --- Page content ---
st.title("Crop Recommendation Dashboard")

prediction_count = get_prediction_count(user)

tab1, tab2 = st.tabs(["Make Prediction", f"History"])

# -----------------------------------------------------------
# TAB 1: PREDICTION FORM
# -----------------------------------------------------------
with tab1:
    st.markdown("Enter the soil and environmental parameters below to get a crop recommendation.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🌱 Soil Nutrients")
            n = st.number_input(
                "Nitrogen (N)  •  0  – 155 mg/kg",
                value=50,
                help="Valid range: 0 to 155mg/kg"
            )
            p = st.number_input(
                "Phosphorus (P)  •  4 – 140 mg/kg",
                value=50,
                help="Valid range: 4 to 140 mg/kg"
            )
            k = st.number_input(
                "Potassium (K)  •  4 – 235 mg/kg",
                value=50,
                help="Valid range: 4 to 235 mg/kg"
            )

        with col2:
            st.markdown("##### 🌤️ Climate Conditions")
            temp = st.number_input(
                "Temperature  •  8 – 52 °C",
                value=25.0,
                help="Valid range: 8 to 52 °C"
            )
            hum = st.number_input(
                "Humidity  •  13 – 100 %",
                value=60.0,
                help="Valid range: 13 to 100 %"
            )

        with col3:
            st.markdown("##### 🧪 Soil Chemistry & Water")
            ph = st.number_input(
                "Soil pH  •  3.5 – 10",
                value=6.5,
                help="Valid range: 3.5 to 10"
            )
            rain = st.number_input(
                "Rainfall  •  17 – 850 mm",
                value=100.0,
                help="Valid range: 17 to 850 mm"
            )

        st.markdown("")
        submit = st.form_submit_button(
            "Get Recommendation",
            width = "stretch",
            type="primary"
        )

    # --- Handle submission ---
    if submit:
        try:
            with st.spinner("Analyzing soil and weather conditions..."):
                time.sleep(1)
                # predict_crop returns (crop, confidence, proba)
                # confidence is already calibrated (Platt scaling) and penalized
                crop, confidence, proba = predict_crop(n, p, k, temp, hum, ph, rain)

            # Get warnings (for display only, does not affect confidence)
            warnings_list = warn_inputs(n, p, k, temp, hum, ph, rain)

            # Show warnings
            if warnings_list:
                st.markdown("#### ⚠️ Heads up:")
                for w in warnings_list:
                    st.warning(w)

            st.success("✅ Prediction Complete")

            # Sort proba for top-2 check (using raw probabilities, not calibrated confidence)
            top = sorted(proba.items(), key=lambda x: x[1], reverse=True)[:2]

            # Show double prediction when the runner-up is genuinely competitive:
            #   - runner-up raw probability > 30% (not just noise)
            #   - gap between 1st and 2nd < 35% (not a dominant single prediction)
            second_exists = (
                len(top) > 1
                and top[1][1] > 0.30
                and (top[0][1] - top[1][1]) < 0.35
            )

            if second_exists:
                # Genuinely ambiguous — surface both crops
                st.info(f"🌾 Primary recommendation: **{top[0][0].upper()}** ({confidence*100:.0f}% confidence)")
                st.warning(f"Also consider: **{top[1][0].upper()}** ({top[1][1]*100:.0f}% probability) — conditions overlap or input values are unusual")

            else:
                # Single prediction – display final calibrated confidence
                st.info(f"🌾 The most suitable crop for these conditions is: **{crop.upper()}**")
                st.caption(f"Confidence: {confidence * 100:.1f}%")

            # Save to history
            if save_prediction_to_history(user, n, p, k, temp, hum, ph, rain, crop):
                st.success("✅ Prediction saved to history!")
            else:
                st.warning("⚠️ Prediction made but could not save to history.")

        except ValueError as e:
            # Hard validation errors — prediction blocked
            error_lines = str(e).replace("Invalid input values:\n", "").split("\n")
            st.markdown("#### ❌ Please fix the following errors:")
            for line in error_lines:
                st.error(line.replace("• ", ""))

# -----------------------------------------------------------
# TAB 2: PREDICTION HISTORY 
# -----------------------------------------------------------
with tab2:
    history = get_user_prediction_history(user)

    if history:
        df_history = pd.DataFrame(history)

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("📊 Total Predictions", len(history))
        with stat_col2:
            crops = df_history["predicted_crop"].nunique()
            st.metric("🌾 Unique Crops", crops)
        with stat_col3:
            most_common = df_history["predicted_crop"].mode()[0].title()
            st.metric("🏆 Most Predicted", most_common)

        st.markdown("")

        search_col, sort_col = st.columns([2, 1])
        with search_col:
            search = st.text_input("🔍 Search crop...", placeholder="Rice, Wheat, etc")
        with sort_col:
            sort_by = st.selectbox("Sort", ["Newest", "Oldest"])

        filtered = history.copy()
        if search:
            filtered = [
                record for record in filtered
                if search.lower() in record["predicted_crop"].lower()
            ]

        filtered = sorted(
            filtered,
            key=lambda x: pd.to_datetime(x["created_at"]),
            reverse=(sort_by == "Newest")
        )

        if filtered:
            for record in filtered:
                date_obj  = pd.to_datetime(record["created_at"])
                date_str  = date_obj.strftime("%b %d, %Y")
                time_str  = date_obj.strftime("%H:%M")
                crop_name = record["predicted_crop"].title()

                with st.expander(f"🌾 {crop_name} — {date_str} at {time_str}", expanded=False):
                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.markdown("**🌱 Soil Nutrients**")
                        st.write(f"• Nitrogen (N): {record['nitrogen']:.1f} mg/kg")
                        st.write(f"• Phosphorus (P): {record['phosphorus']:.1f} mg/kg")
                        st.write(f"• Potassium (K): {record['potassium']:.1f} mg/kg")

                        st.markdown("**🧪 Soil Chemistry**")
                        st.write(f"• pH Level: {record['ph']:.2f}")

                    with detail_col2:
                        st.markdown("**🌤️ Environment**")
                        st.write(f"• Temperature: {record['temperature']:.1f} °C")
                        st.write(f"• Humidity: {record['humidity']:.1f} %")
                        st.write(f"• Rainfall: {record['rainfall']:.1f} mm")

                        st.markdown("**📊 Summary**")
                        npk_total = record['nitrogen'] + record['phosphorus'] + record['potassium']
                        st.write(f"• Total NPK: {npk_total:.1f}")

                    st.markdown("---")
                    del_col, _ = st.columns([1, 4])
                    with del_col:
                        if st.button("🗑️ Delete", key=f"delete_{record['id']}", width = "stretch"):
                            if delete_prediction(record['id']):
                                st.success("Deleted!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to delete.")
        else:
            st.warning("No predictions match your search.")

    else:
        st.info("📭 No prediction history yet. Make a prediction to get started!")
