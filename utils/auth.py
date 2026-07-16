import streamlit as st
from utils.supabase_client import supabase
from streamlit_cookies_manager import EncryptedCookieManager

# -------------------------------
# Cookie Manager Setup
# -------------------------------

cookies = EncryptedCookieManager(
    prefix="agrotree_",
    password=st.secrets["COOKIE_PASSWORD"],
)

if not cookies.ready():
    # Cookies aren't loaded yet on first run; stop and let Streamlit rerun.
    st.stop()



# Get user email safely
def get_user_email(user):
    """Safely extract email from Supabase user object"""

    if user is None:
        return None

    # Direct email field
    if hasattr(user, "email") and user.email:
        return user.email

    # Sometimes stored inside metadata
    if hasattr(user, "user_metadata") and isinstance(user.user_metadata, dict):
        email = user.user_metadata.get("email")
        if email:
            return email

    return "User"


# -------------------------------
# Get user fullname
# -------------------------------
def get_user_fullname(user):
    """Get user fullname from session cache or user object."""

    # Try session cache first (FASTEST - no DB call)
    if "user_fullname" in st.session_state and st.session_state["user_fullname"]:
        return st.session_state["user_fullname"]

    # Try to fetch from database if not cached
    if user:
        _cache_user_fullname(user)
        if st.session_state.get("user_fullname"):
            return st.session_state["user_fullname"]

    # Fallback to email or default
    if user and hasattr(user, "email"):
        return user.email.split("@")[0].title()

    return "User"


# -------------------------------
# Cache user fullname from database
# -------------------------------
def _cache_user_fullname(user):
    """Fetch fullname from profiles table and cache in session."""
    if not user:
        return

    try:
        # Query profiles table for fullname
        res = supabase.table("profiles").select("fullname").eq("id", user.id).single().execute()
        if res.data and "fullname" in res.data:
            st.session_state["user_fullname"] = res.data["fullname"]
    except Exception:
        # Fallback to email if fullname not found
        st.session_state["user_fullname"] = user.email.split("@")[0].title() if user.email else "User"


# -------------------------------
# Save tokens to encrypted browser cookies
# -------------------------------
def _save_session_cookies(session_data):
    """Persist access/refresh tokens in encrypted browser cookies."""

    try:
        cookies["access_token"] = session_data.get("access_token") or ""
        cookies["refresh_token"] = session_data.get("refresh_token") or ""
        cookies.save()
    except Exception as e:
        print("Cookie save error:", e)


# -------------------------------
# Clear session cookies
# -------------------------------
def _clear_session_cookies():
    """Remove cached tokens from browser cookies."""

    try:
        cookies["access_token"] = ""
        cookies["refresh_token"] = ""
        cookies.save()
    except Exception as e:
        print("Cookie clear error:", e)


# -------------------------------
# Initialize session with persistence
# -------------------------------
def init_session():
    """Initialize session state with automatic persistence and recovery via cookies."""

    # Initialize session state variables
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if "session" not in st.session_state:
        st.session_state["session"] = None

    if "user_fullname" not in st.session_state:
        st.session_state["user_fullname"] = None

    # If user already set, skip
    if st.session_state["user"]:
        return

    # Try to recover active session from Supabase (in-memory client state)
    try:
        session = supabase.auth.get_session()

        if session:
            user_res = supabase.auth.get_user()

            if user_res and user_res.user:
                st.session_state["user"] = user_res.user
                st.session_state["session"] = session
                _cache_user_fullname(user_res.user)
                return

    except Exception as e:
        print("Current session error:", e)

    # Try to restore session from encrypted browser cookies
    access_token = cookies.get("access_token")
    refresh_token = cookies.get("refresh_token")

    if not access_token or not refresh_token:
        return

    try:
        auth_res = supabase.auth.set_session(access_token, refresh_token)

        if auth_res and auth_res.user:
            st.session_state["user"] = auth_res.user
            st.session_state["session"] = auth_res.session

            # Supabase may rotate the refresh token; persist the latest ones
            _save_session_cookies({
                "access_token": auth_res.session.access_token,
                "refresh_token": auth_res.session.refresh_token,
            })

            _cache_user_fullname(auth_res.user)

    except Exception as e:
        print("Session restore error:", e)
        _clear_session_cookies()


# -------------------------------
# Login user
# -------------------------------
def login_user(email, password):
    """Authenticate user with email and password."""

    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        st.session_state["user"] = res.user
        st.session_state["session"] = res.session

        # Persist tokens in encrypted browser cookies
        _save_session_cookies({
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
        })

        _cache_user_fullname(res.user)

        return True

    except Exception as e:
        print("Login error:", e)
        return False


# -------------------------------
# Sign up user
# -------------------------------
def signup_user(fullname, email, password):
    """Register new user with email, password, and full name."""

    try:
        # Create auth account
        auth_res = supabase.auth.sign_up({
            "email": email,
            "password": password
        })

        if not auth_res.user:
            return False

        # Insert profile data
        supabase.table("profiles").insert({
            "id": auth_res.user.id,
            "fullname": fullname,
            "email": email
        }).execute()

        # Cache fullname in session
        st.session_state["user_fullname"] = fullname

        return True

    except Exception as e:
        print("Signup error:", e)
        return False


# -------------------------------
# Logout user
# -------------------------------
def logout_user():
    """Clear session and sign out from Supabase."""

    try:
        supabase.auth.sign_out()

    except Exception as e:
        print("Logout error:", e)

    st.session_state["user"] = None
    st.session_state["session"] = None
    st.session_state["user_fullname"] = None

    _clear_session_cookies()


@st.dialog("Confirm Logout")
def confirm_logout_dialog():
    st.write("Are you sure you want to log out?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Cancel", width="stretch"):
            st.rerun()

    with col2:
        if st.button("Logout", width="stretch", type="primary"):
            logout_user()
            st.rerun()


# -------------------------------
# Require authentication
# -------------------------------
def require_auth():
    """Redirect to login page if user is not authenticated."""

    if not st.session_state.get("user"):
        st.switch_page("pages/login.py")