import streamlit as st

from utils.auth import init_session, is_admin, require_auth, logout_user
from admin import retrain_model, view_report


st.set_page_config(page_title="Admin Panel", layout="wide")

init_session()
require_auth()

if not is_admin():
    st.error("Access Denied! Admin only.")
    st.stop()

st.title("Admin")
col_left, col_right = st.columns([3, 1])
with col_right:
    if st.button("Logout", type="secondary"):
        logout_user()
        st.rerun()

sections = {
    "📈 View Report": view_report.render,
    "🔄 Retrain Model": retrain_model.render,
}

selected_section = st.radio("Choose an admin section", list(sections.keys()), horizontal=True)
st.divider()
sections[selected_section]()