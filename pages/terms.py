import streamlit as st

st.set_page_config(page_title="Terms of Service", layout="centered")
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
st.page_link("app.py", label="Back to Home", icon=":material/arrow_back:")

# Top Left Back Arrow using Native Streamlit Material Icon string
st.title("Terms of Service")
st.markdown("AgroTree – Crop Recommendation System")

st.markdown("""
### 1. Educational Purpose
AgroTree is developed as an academic project for demonstration and learning.

### 2. No Guarantee of Accuracy
Recommendations are generated using a Decision Tree algorithm.
Accuracy depends on dataset quality and input values.

### 3. User Responsibility
Users must provide accurate input data.
Final farming decisions remain the responsibility of the user.

### 4. Limitation of Liability
Developers are not responsible for agricultural or financial losses.

### 5. System Availability
System availability may vary as it is hosted on free-tier cloud services.
""")
