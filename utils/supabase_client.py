import streamlit as st
from supabase import create_client, Client

@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

supabase: Client = get_supabase()


@st.cache_resource
def get_admin_supabase(service_key: str | None):
    if not service_key:
        return None

    return create_client(
        st.secrets["SUPABASE_URL"],
        service_key
    )


admin_supabase = get_admin_supabase(st.secrets.get("SUPABASE_SERVICE_KEY"))