import streamlit as st
from pages.heart_disease_detection import show_heart_disease_detection
from pages.healthcare_trends import show_healthcare_trends
from pages.exploratory_analysis import show_exploratory_analysis

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: white;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: #2E8B57;
    }
    .stMarkdown {
        color: #2E8B57;
    }
    .beautiful-title {
        background: linear-gradient(90deg, #2E8B57, #4CAF50, #45a049);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .page-indicator {
        text-align: center;
        color: #2E8B57;
        font-size: 1.1rem;
        margin: 10px 0;
        padding: 5px;
        background: linear-gradient(90deg, #f0f8f0, #e8f5e8);
        border-radius: 10px;
    }
    .github-link {
        text-align: center;
        margin: 20px 0;
        padding: 15px;
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        border: 2px solid #dee2e6;
    }
    .github-link a {
        color: #2E8B57;
        text-decoration: none;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .github-link a:hover {
        color: #4CAF50;
        text-decoration: underline;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #2E8B57, #4CAF50);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with GitHub link
with st.sidebar:
    st.markdown("## 🔗 Project Links")
    st.markdown("""
    <div class="github-link">
        <a href="https://github.com/Ahmed-Amr777/hearthealth" target="_blank">
            📂 View on GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    This is a comprehensive machine learning project for heart disease prediction using the Cleveland Heart Disease dataset.
    
    **Features:**
    - Data preprocessing & feature selection
    - Multiple ML algorithms (SVM, Random Forest, XGBoost)
    - Interactive web interface
    - Real-time predictions
    """)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'detection'

# Beautiful centered title
st.markdown('<h1 class="beautiful-title">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)

# Navigation
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("← Back"):
        if st.session_state.current_page == 'trends':
            st.session_state.current_page = 'detection'
        elif st.session_state.current_page == 'analysis':
            st.session_state.current_page = 'trends'
        st.rerun()

with col2:
    # Page indicator with better styling
    page_names = {
        'detection': '📊 Heart Disease Detection',
        'trends': '📈 Healthcare Trends', 
        'analysis': '🔍 Exploratory Data Analysis'
    }
    current_page_name = page_names.get(st.session_state.current_page, '')
    page_number = {'detection': 1, 'trends': 2, 'analysis': 3}[st.session_state.current_page]
    st.markdown(f'<div class="page-indicator">Page {page_number} of 3: {current_page_name}</div>', unsafe_allow_html=True)

with col3:
    if st.button("Next →"):
        if st.session_state.current_page == 'detection':
            st.session_state.current_page = 'trends'
        elif st.session_state.current_page == 'trends':
            st.session_state.current_page = 'analysis'
        st.rerun()

# Display current page
if st.session_state.current_page == 'detection':
    show_heart_disease_detection()
elif st.session_state.current_page == 'trends':
    show_healthcare_trends()
else:
    show_exploratory_analysis()

# Footer with GitHub link
st.markdown("""
<div class="footer">
    <a href="https://github.com/Ahmed-Amr777/hearthealth" target="_blank" style="color: white; text-decoration: none;">
        🔗 View Source Code on GitHub | Heart Disease Prediction Project
    </a>
</div>
""", unsafe_allow_html=True)