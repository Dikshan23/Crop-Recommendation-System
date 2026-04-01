import streamlit as st
from utils.auth import require_auth, logout_user, init_session, get_user_email

st.set_page_config(page_title="Dashboard", layout="wide")

# Initialize session
init_session()
require_auth()  # Will redirect to login if not logged in

# Sidebar logout
with st.sidebar:
    user_email = get_user_email(st.session_state['user'])
    st.write(f"👤 **{user_email}**")
    if st.button("🚪 Logout", use_container_width=True):
        logout_user()
        st.switch_page("pages/login.py")

st.title("🌾 Farmer Dashboard - Crop Recommendation")