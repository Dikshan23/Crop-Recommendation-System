import streamlit as st
from utils.auth import signup_user, init_session
# Removed emoji, added collapsed sidebar state
st.set_page_config(page_title="Sign Up - AgroTree", layout="wide", initial_sidebar_state="collapsed")

init_session()
st.markdown("""
<style>
    /* Import identical Google Fonts & Icons */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Poppins:wght@600;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)



# Top Left Back Arrow using Native Streamlit Material Icon string
st.page_link("app.py", label="Back to Home", icon=":material/arrow_back:")

st.write("") 
st.write("") 

# Center the signup form
_, col_center, _ = st.columns([1, 1.5, 1])

with col_center:
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Create an Account</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; margin-bottom: 30px;'>Join AgriAI to start optimizing your crop yield today.</p>", unsafe_allow_html=True)
    
    with st.form("signup_form"):
        email = st.text_input("Email Address", placeholder="e.g., farmer@agriai.com")
        password = st.text_input("Password", type="password", placeholder="Create a strong password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
        
        st.write("")
        submit = st.form_submit_button("Sign Up", type="primary", use_container_width=True)
        
        if submit:
           if password != confirm_password:
               st.error("Password do not match")
           elif signup_user(email, password):
               st.success("Account created ! Please log in.")
               st.switch_page("pages/login.py")
           else:
               st.error("Siggnup failed.")
            
               
    st.write("")
    st.markdown("<p style='text-align: center; color: gray;'>Already have an account?</p>", unsafe_allow_html=True)
    
    if st.button("Log In Instead", icon=":material/login:", use_container_width=True):
        st.switch_page("pages/login.py")
