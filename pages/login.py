import streamlit as st
from utils.auth import login_user, init_session

st.set_page_config(page_title="Log In - AgroTree", layout="wide", initial_sidebar_state="collapsed")

init_session()

# If already logged in, redirect to dashboard
if st.session_state.get("user"):
    st.switch_page("pages/dashboard.py")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Poppins:wght@600;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# Back to home
st.page_link("app.py", label="Back to Home", icon=":material/arrow_back:")

st.write("")
st.write("")

# Center the login form
_, col_center, _ = st.columns([1, 1.5, 1])

with col_center:
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Welcome Back</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; margin-bottom: 30px;'>Log in to your AgroTree account</p>", unsafe_allow_html=True)
    
    with st.form("login_form"):
        email = st.text_input("Email Address", placeholder="e.g., farmer@agrotree.com")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        st.write("")
        submit = st.form_submit_button("Log In", type="primary", use_container_width=True)
        
        if submit:
            if not email or not password:
                st.error("Please enter both email and password")
            elif login_user(email, password):
                st.success("Login successful! Redirecting...")
                st.switch_page("pages/dashboard.py")
            else:
                st.error("Invalid email or password")
    
    st.write("")
    st.markdown("<p style='text-align: center; color: gray;'>Don't have an account?</p>", unsafe_allow_html=True)
    
    if st.button("Sign Up Instead", icon=":material/person_add:", use_container_width=True):
        st.switch_page("pages/signup.py")