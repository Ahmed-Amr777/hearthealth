import streamlit as st

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
        width: 100%;
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

# Beautiful centered title
st.markdown('<h1 class="beautiful-title">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)

# Main content
st.markdown("## Welcome to the Heart Disease Prediction System")

st.markdown("""
This comprehensive machine learning project uses the Cleveland Heart Disease dataset to predict heart disease risk.

### 🚀 Features:
- **Heart Disease Detection**: Interactive prediction interface
- **Healthcare Trends**: Statistical analysis and insights
- **Exploratory Data Analysis**: Deep dive into the dataset

### 📊 Dataset Information:
The system uses the Cleveland Heart Disease dataset with the following features:
- Age, Sex, Chest Pain Type
- Blood Pressure, Cholesterol, Blood Sugar
- ECG Results, Heart Rate, Exercise Angina
- ST Depression, Slope, Vessel Count, Thalassemia

### 🔗 Navigation:
Use the sidebar to navigate between different pages and explore the system's capabilities.
""")

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

# Footer with GitHub link
st.markdown("""
<div class="footer">
    <a href="https://github.com/Ahmed-Amr777/hearthealth" target="_blank" style="color: white; text-decoration: none;">
        🔗 View Source Code on GitHub | Heart Disease Prediction Project
    </a>
</div>
""", unsafe_allow_html=True)