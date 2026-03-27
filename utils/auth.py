import streamlit as st
from utils.supabase_client import supabase
import json
from pathlib import Path

# Session cache file for persistence across server restarts
CACHE_DIR = Path.home() / ".agrotree_cache"
CACHE_DIR.mkdir(exist_ok=True)
SESSION_CACHE_FILE = CACHE_DIR / "session_cache.json"

# Get user email safely from UserResponse object
def get_user_email(user):
    """Safely extract email from Supabase user object"""
    if user is None:
        return None
    # Try different possible locations for email
    if hasattr(user, 'email') and user.email:
        return user.email
    if hasattr(user, 'user_metadata') and isinstance(user.user_metadata, dict):
        return user.user_metadata.get('email')
    # Fallback to ID if email not available
    return getattr(user, 'id', 'User')

# Save session to cache file
def _save_session_cache(session_data):
    """Save session refresh token to local cache"""
    try:
        cache_data = {
            "refresh_token": session_data.get("refresh_token"),
            "access_token": session_data.get("access_token")
        }
        with open(SESSION_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print("Cache save error:", e)

# Load session from cache file
def _load_session_cache():
    """Load session refresh token from local cache"""
    try:
        if SESSION_CACHE_FILE.exists():
            with open(SESSION_CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print("Cache load error:", e)
    return None

# Clear session cache
def _clear_session_cache():
    """Clear the session cache file"""
    try:
        if SESSION_CACHE_FILE.exists():
            SESSION_CACHE_FILE.unlink()
    except Exception as e:
        print("Cache clear error:", e)

# Initialize session with persistent login support across server restarts
def init_session():
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "session" not in st.session_state:
        st.session_state["session"] = None
    
    # Try to recover session from Supabase
    if st.session_state["user"] is None:
        try:
            # First, try current session in memory
            session = supabase.auth.get_session()
            if session:
                user = supabase.auth.get_user()
                if user:
                    st.session_state["user"] = user
                    st.session_state["session"] = session
                    return
        except Exception as e:
            print("Current session error:", e)
        
        # If no current session, try to recover from cache using refresh token
        try:
            cached = _load_session_cache()
            if cached and cached.get("refresh_token"):
                # Refresh the session using the stored refresh token
                session = supabase.auth.refresh_session(cached["refresh_token"])
                if session and session.user:
                    st.session_state["user"] = session.user
                    st.session_state["session"] = session
                    # Update cache with new tokens
                    _save_session_cache({
                        "refresh_token": session.session.refresh_token,
                        "access_token": session.session.access_token
                    })
        except Exception as e:
            print("Session refresh error:", e)
            # If refresh fails, clear the cache
            _clear_session_cache()
# LOGIN 
def login_user(email, password):
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        st.session_state["user"] = res.user
        st.session_state["session"] = res.session
        
        # Save session to cache for persistence across restarts
        session_data = {
            "refresh_token": res.session.refresh_token,
            "access_token": res.session.access_token
        }
        _save_session_cache(session_data)

        return True
    except Exception as e:
        print("Login error:", e)
        return False
    
# SIGN UP
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
    
# LOGOUT 
def logout_user():
    try:
        supabase.auth.sign_out()
    except Exception as e:
        print("Logout error:", e)
    
    st.session_state["user"] = None
    st.session_state["session"] = None    
    # Clear cached session
    _clear_session_cache()
# Protect Page
def require_auth():
    if not st.session_state.get("user"):
        st.switch_page("pages/login.py")
