import streamlit as st
from utils.auth import signup_user, init_session
import re
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
    .stForm{
        padding : 25px;
        border-radius : 14px;
        border: 1px solif #e0e0e0;
        background-color:white;
    }
    .password-rules{
        font-size : 14px;
        color: red;
        margin-top:-10px;
        margin-bottom:10px;
    }
</style>
""", unsafe_allow_html=True)

def is_valid_fullname(fullname):
    pattern = r"^[A-Xa-z]+(?: [A-Za-z]+)*$"
    return re.match(pattern, fullname)

#Validation Functions
def is_valid_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email)

def validate_password(password):
    errors = []

    if len(password)<8:
        errors.append("Password must be atleast 8 characters long.")
    
    if not re.search(r"[A-Z]",password):
        errors.append("Password must contain atleast one uppercase letter.")

    if not re.search(r"[a-z]",password):
        errors.append("Password must contain atleast one uppercase letter.")

    if not re.search(r"\d", password):
        errors.append("Password must contain atleast one number.")
    
    if not re.search(r"[!@$$%^&*(),.?\":{}|<>]", password):
        errors.append("Password must contain atleast one special character.")

    return errors


#Navigation
st.page_link("app.py", label="Back to Home", icon=":material/arrow_back:")

st.write("") 
st.write("") 

#layout
_, col_center, _ = st.columns([1, 1.5, 1])

with col_center:
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Create an Account</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; margin-bottom: 30px;'>Join AgroTree to start optimizing your crop yield today.</p>", unsafe_allow_html=True)
    
    with st.form("signup_form"):
        fullname = st.text_input("Full Name", placeholder = "Full Name")
        email = st.text_input("Email Address", placeholder="e.g., farmer@agro.com")
        password = st.text_input("Password", type="password", placeholder="Create a strong password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
        
        st.write("")

        submit = st.form_submit_button("Sign Up", type="primary", width = "stretch")
        
        st.write("")

        if submit:
           
           if not email or not password or not confirm_password:
               st.error("All fields are required.")
           elif not is_valid_email(email):
               st.error("Please enter a valid email address")
           else:
               password_errors = validate_password(password)

               if password_errors:
                   for error in password_errors:
                       st.error(error)
               elif password != confirm_password:
                   st.error("Passwords do not match.")
               else:
                   success = signup_user(fullname.strip(), email.strip(), password)

                   if success:
                       st.success("Account created successfully.")
                       st.switch_page("pages/login.py")
                   else:
                       st.error("Signup failed. Email may already exist.")             
               
    st.write("")
    st.markdown("<p style='text-align: center; color: gray;'>Already have an account?</p>", unsafe_allow_html=True)
    
    if st.button("Log In", icon=":material/login:", width = "stretch"):
        st.switch_page("pages/login.py")