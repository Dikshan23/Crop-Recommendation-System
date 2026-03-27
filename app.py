import streamlit as st
import graphviz
from PIL import Image
from utils.auth import init_session

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AgroTree - Crop Recommendation System", layout="wide", initial_sidebar_state="collapsed")

# Initialize session and attempt to restore persistent login
init_session()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Poppins:wght@600;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
    }

    .flex-center {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }

    .feature-card {
        background-color: var(--secondary-background-color);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid var(--faded-text-color);
        height: 300px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(76, 175, 80, 0.15);
        border-color: #4CAF50;
    }
    .icon-large {
        font-size: 50px !important;
        color: #4CAF50; 
        margin-bottom: 15px;
    }

    .dev-card {
        text-align: center;
        padding: 35px 20px;
        background-color: var(--secondary-background-color);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    .dev-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 35px rgba(76, 175, 80, 0.2);
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    .dev-img {
        width: 130px;
        height: 130px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 20px;
        border: 4px solid #4CAF50;
        padding: 4px;
        background-color: white;
    }
    .dev-name {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 8px;
        color: var(--text-color);
    }
    .roll {
        display: inline-block;
        background-color: rgba(76, 175, 80, 0.15);
        color: #2E7D32;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .cta-container {
        background: linear-gradient(135deg, #1B5E20 0%, #4CAF50 100%);
        border-radius: 20px;
        padding: 60px 20px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(76, 175, 80, 0.3);
        margin-bottom: 20px;
    }
    .cta-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 15px;
        line-height: 1.2;
    }
    .cta-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9) !important;
        margin-bottom: 0;
        font-weight: 400;
    }

    .section-spacing {
        margin-top: 100px;
        margin-bottom: 50px;
    }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
hero_col1, hero_col2 = st.columns([1.2, 1], gap="large")

with hero_col1:
    st.markdown("<h1 style='font-size: 3.5rem; line-height: 1.2;'>AgroTree<br><span style='color: #4CAF50;'>Crop Recommendation System</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: gray;'>A Decision Tree based system that analyzes soil nutrients (N, P, K, pH) and environmental conditions (temperature, humidity, rainfall) to recommend the most suitable crop.</p>", unsafe_allow_html=True)
    
    st.write("")
    btn_col1, btn_col2, _ = st.columns([1, 1, 1])
    with btn_col1:
        if st.button("Get Started", icon=":material/eco:", type="primary", use_container_width=True):
            st.switch_page("pages/login.py")
    with btn_col2:
        st.button("Learn More", icon=":material/menu_book:", use_container_width=True)

with hero_col2:
    img = Image.open("images/hero.png")
    st.image(img, use_container_width=True)

# --- FEATURES SECTION ---
st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Why Choose AgroTree?</h2><br>", unsafe_allow_html=True)

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    <div class="feature-card">
        <span class="material-symbols-outlined icon-large">science</span>
        <h3>Scientific Soil Analysis</h3>
        <p>Enter Nitrogen (N), Phosphorus (P), Potassium (K), and pH values to generate accurate crop predictions based on agricultural dataset.</p>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class="feature-card">
        <span class="material-symbols-outlined icon-large">account_tree</span>
        <h3>CART Decision Tree Model</h3>
        <p>Uses a CART Decision Tree algorithm implemented from scratch to classify and recommend the most suitable crop.</p>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class="feature-card">
        <span class="material-symbols-outlined icon-large">cloud</span>
        <h3>Climate-Based Recommendation</h3>
        <p>Analyzes user-provided temperature, humidity, and rainfall values for accurate crop prediction.</p>
    </div>
    """, unsafe_allow_html=True)

# --- HOW IT WORKS SECTION (ORIGINAL LAYOUT KEPT) ---
st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>How It Works</h2><br>", unsafe_allow_html=True)

hw_col1, hw_col2 = st.columns([1.2, 1], gap="large")

with hw_col1:
    st.markdown("<h4 class='flex-center' style='margin-bottom: 20px;'><span class='material-symbols-outlined'>play_circle</span> System Demonstration</h4>", unsafe_allow_html=True)
    st.video("https://www.youtube.com/watch?v=A2dY2W_q72I")

with hw_col2:
    st.markdown("<h4 class='flex-center' style='margin-bottom: 20px;'><span class='material-symbols-outlined'>account_tree</span> System Flow Architecture</h4>", unsafe_allow_html=True)
    
    flowchart = graphviz.Digraph()
    flowchart.attr(rankdir='TB', bgcolor='transparent', size='5,5') 
    
    flowchart.attr('node', shape='box', style='rounded,filled', fillcolor='#e8f5e9', color='#4CAF50', fontname='Inter', fontsize='12', height='0.6', width='4')
    flowchart.attr('edge', color='#4CAF50', penwidth='2', arrowsize='0.8')
    
    flowchart.node('A', '1. User Inputs Soil & Climate Data\n(N, P, K, pH, Temp, Humidity, Rain)')
    flowchart.node('B', '2. Input Validation\n(Parameter Range Checking)')
    flowchart.node('C', '3. CART Decision Tree Algorithm\n(Implemented from Scratch)')
    flowchart.node('D', '4. Output: Recommended Crop')
    
    flowchart.edges([('A', 'B'), ('B', 'C'), ('C', 'D')])
    
    st.graphviz_chart(flowchart, use_container_width=True)

# --- DEVELOPERS SECTION ---
st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Meet the Developers</h2><br>", unsafe_allow_html=True)

dev_col1, dev_col2, dev_col3 = st.columns(3)

with dev_col1:
    st.markdown("""
    <div class="dev-card">
        <img src="" class="dev-img">
        <div class="dev-name">Dikshan KC</div>
        <div class="roll">79010772</div>
    </div>
    """, unsafe_allow_html=True)

with dev_col2:
    st.markdown("""
    <div class="dev-card">
        <img src="" class="dev-img">
        <div class="dev-name">Prajwal Pathak</div>
        <div class="roll">79010785</div>
    </div>
    """, unsafe_allow_html=True)

with dev_col3:
    st.markdown("""
    <div class="dev-card">
        <img src="" class="dev-img">
        <div class="dev-name">Chetan Bhattarai</div>
        <div class="roll">79010771</div>
    </div>
    """, unsafe_allow_html=True)

# --- CTA SECTION ---
st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)

st.markdown("""
<div class='cta-container'>
    <h2 class='cta-title'>Start Data-Driven Crop Planning</h2>
    <p class='cta-subtitle'>Register and receive crop recommendations based on scientific soil and climate analysis.</p>
</div>
""", unsafe_allow_html=True)

_, cta_col, _ = st.columns([1, 1, 1])
with cta_col:
    if st.button("Get Started Now", icon=":material/rocket_launch:", type="primary", use_container_width=True):
        st.switch_page("pages/login.py")

# --- FOOTER ---
st.markdown("<div style='margin-top: 50px;'></div><hr>", unsafe_allow_html=True)
footer_col1, footer_col2 = st.columns([3, 1])
with footer_col1:
    st.markdown("<p style='color: gray; font-size: 0.9rem;'>© 2026 AgroTree – Crop Recommendation System<br>Orchid International College</p>", unsafe_allow_html=True)
with footer_col2:
   st.page_link("pages/privacy.py", label="Privacy Policy")
   st.page_link("pages/terms.py", label="Terms of Service")