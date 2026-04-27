import streamlit as st
import keyring
from utils.supabase_client import supabase

# Config
SERVICE_NAME = "agrotree_supabase"
REFRESH_KEY = "refresh_token"

# Save token securely
def _save_refresh_token(token: str):
    keyring.set_password(SERVICE_NAME, REFRESH_KEY, token)

# Load token
def _load_refresh_token():
    return keyring.get_password(SERVICE_NAME, REFRESH_KEY)

# Delete token
def _clear_refresh_token():
    keyring.delete_password(SERVICE_NAME, REFRESH_KEY)

# Initialize session
def init_session():

    if "user" not in st.session_state:
        st.session_state["user"] = None

    if "session" not in st.session_state:
        st.session_state["session"] = None

    # Already logged in in current runtime
    if st.session_state["user"]:
        return

    # Try Supabase session first (if still alive in memory)
    try:
        session_res = supabase.auth.get_session()

        if session_res and session_res.session:
            user_res = supabase.auth.get_user()

            if user_res and user_res.user:
                st.session_state["user"] = user_res.user
                st.session_state["session"] = session_res.session
                return

    except Exception:
        pass

    # Fallback: restore from secure OS storage
    try:
        refresh_token = _load_refresh_token()

        if refresh_token:

            session_res = supabase.auth.refresh_session(refresh_token)

            if session_res and session_res.session:

                st.session_state["user"] = session_res.user
                st.session_state["session"] = session_res.session

                # update stored token (rotation)
                _save_refresh_token(session_res.session.refresh_token)

    except Exception:
        _clear_refresh_token()

# Login user
def login_user(email, password):

    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        st.session_state["user"] = res.user
        st.session_state["session"] = res.session

        # store refresh token securely
        _save_refresh_token(res.session.refresh_token)

        return True

    except Exception:
        st.error("Login failed")
        return False

# Sign up user
def signup_user(email, password):

    try:
        supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        return True

    except Exception:
        st.error("Signup failed")
        return False

# Logout user
def logout_user():

    try:
        supabase.auth.sign_out()
    except Exception:
        pass

    st.session_state.clear()

    try:
        _clear_refresh_token()
    except Exception:
        pass

    st.rerun()

# Require authentication
def require_auth():

    if not st.session_state.get("user"):
        st.switch_page("pages/login.py")

# Get user email
def get_user_email(user):
    """Extract email from user object"""
    if user and hasattr(user, 'email'):
        return user.email
    return None