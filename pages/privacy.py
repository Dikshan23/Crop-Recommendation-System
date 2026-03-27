import streamlit as st

st.set_page_config(page_title="Privacy Policy", layout="centered")
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
st.title("Privacy Policy")
st.markdown("AgroTree – Crop Recommendation System")

st.markdown("""
### 1. Information We Collect
- Email address (for authentication)
- Soil parameters (N, P, K, pH)
- Climate parameters (temperature, humidity, rainfall)
- Prediction history linked to user account

### 2. Purpose of Data Collection
Data is used only for:
- User authentication
- Generating crop recommendations
- Storing prediction history

### 3. Data Storage
- Data is stored securely using Supabase PostgreSQL.
- Passwords are encrypted using Supabase authentication.
- No financial or sensitive personal data is collected.

### 4. Data Sharing
We do not sell or share user data with third parties.

### 5. Academic Disclaimer
This system is developed for academic purposes and should not replace professional agricultural consultation.
""")