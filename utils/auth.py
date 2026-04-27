import streamlit as st
from utils.supabase_client import supabase
import json
from pathlib import Path

# Session cache path
CACHE_DIR = Path.home() / ".agrotree_cache"
CACHE_DIR.mkdir(exist_ok=True)
SESSION_CACHE_FILE = CACHE_DIR / "session_cache.json"


# Get user email
def get_user_email(user):
    """Safely extract email from Supabase user object"""

    if user is None:
        return None

    # Check direct email field
    if hasattr(user, "email") and user.email:
        return user.email

    # Check metadata email
    if hasattr(user, "user_metadata") and isinstance(user.user_metadata, dict):
        email = user.user_metadata.get("email")
        if email:
            return email

    return "User"


# Save session cache
def _save_session_cache(session_data):
    """Save refresh/access token to cache"""

    try:
        cache_data = {
            "refresh_token": session_data.get("refresh_token"),
            "access_token": session_data.get("access_token")
        }

        with open(SESSION_CACHE_FILE, "w") as f:
            json.dump(cache_data, f)

    except Exception as e:
        print("Cache save error:", e)


# Load session cache
def _load_session_cache():
    """Load session from cache"""

    try:
        if SESSION_CACHE_FILE.exists():
            with open(SESSION_CACHE_FILE, "r") as f:
                return json.load(f)

    except Exception as e:
        print("Cache load error:", e)

    return None


# Clear session cache
def _clear_session_cache():
    """Remove cached session"""

    try:
        if SESSION_CACHE_FILE.exists():
            SESSION_CACHE_FILE.unlink()

    except Exception as e:
        print("Cache clear error:", e)


# Initialize session
def init_session():

    if "user" not in st.session_state:
        st.session_state["user"] = None

    if "session" not in st.session_state:
        st.session_state["session"] = None

    # Skip if user already in session
    if st.session_state["user"]:
        return

    # Try active Supabase session
    try:

        session_res = supabase.auth.get_session()

        if session_res and session_res.session:

            user_res = supabase.auth.get_user()

            if user_res and user_res.user:
                st.session_state["user"] = user_res.user
                st.session_state["session"] = session_res.session
                return

    except Exception as e:
        print("Current session error:", e)

    # Try refresh token from cache
    try:

        cached = _load_session_cache()

        if cached and cached.get("refresh_token"):

            session_res = supabase.auth.refresh_session(
                cached["refresh_token"]
            )

            if session_res and session_res.user:

                st.session_state["user"] = session_res.user
                st.session_state["session"] = session_res.session

                # Update cached tokens
                _save_session_cache({
                    "refresh_token": session_res.session.refresh_token,
                    "access_token": session_res.session.access_token
                })

    except Exception as e:
        print("Session refresh error:", e)
        _clear_session_cache()


# Login user
def login_user(email, password):

    try:

        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        st.session_state["user"] = res.user
        st.session_state["session"] = res.session

        # Cache tokens
        _save_session_cache({
            "refresh_token": res.session.refresh_token,
            "access_token": res.session.access_token
        })

        return True

    except Exception as e:
        print("Login error:", e)
        return False


# Sign up user
def signup_user(email, password):

    try:

        supabase.auth.sign_up({
            "email": email,
            "password": password
        })

        return True

    except Exception as e:
        print("Signup error:", e)
        return False


# Logout user
def logout_user():

    try:
        supabase.auth.sign_out()

    except Exception as e:
        print("Logout error:", e)

    st.session_state["user"] = None
    st.session_state["session"] = None

    _clear_session_cache()


# Redirect unauthenticated users
def require_auth():

    if not st.session_state.get("user"):
        st.switch_page("pages/login.py")