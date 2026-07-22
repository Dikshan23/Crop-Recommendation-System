import streamlit as st

from utils.auth import init_session, is_admin, require_auth
from utils.supabase_client import supabase, admin_supabase


def require_admin_page(page_title: str):
    st.set_page_config(page_title=page_title, layout="wide")
    init_session()
    require_auth()

    if not is_admin():
        st.error("Access Denied! Admin only.")
        st.stop()


def get_admin_db():
    return admin_supabase or supabase


def show_admin_db_warning():
    if admin_supabase is None:
        st.warning("Admin database access is using the regular client because SUPABASE_SERVICE_KEY is not configured.")