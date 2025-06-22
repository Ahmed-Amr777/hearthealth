import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="📊",
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
</style>
""", unsafe_allow_html=True)

# Beautiful title
st.markdown('<h1 class="beautiful-title">📊 Heart Disease Detection</h1>', unsafe_allow_html=True)

st.markdown("""
This page allows you to input patient information and get a heart disease prediction using our trained SVM model.
""")

# Check if model exists
model_path = "models/svm_heart_model.pkl"
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.info("Please ensure the SVM model is trained and saved in the models directory.")
    st.stop()

# Heart Disease Detection Form
with st.form("heart_disease_form"):
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        
        cp_options = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
        cp = st.selectbox("Chest Pain Type", list(cp_options.keys()), format_func=lambda x: cp_options[x])
        
        trestbps = st.number_input("Resting BP (mm Hg)", 90, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        restecg_options = {0: "Normal", 1: "ST-T abnormality", 2: "Left ventricular hypertrophy"}
        restecg = st.selectbox("Resting ECG", list(restecg_options.keys()), format_func=lambda x: restecg_options[x])
        
    with col2:
        thalach = st.number_input("Maximum Heart Rate", 60, 202, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 0.0, step=0.1)
        
        slope_options = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
        slope = st.selectbox("Slope of Peak Exercise ST", list(slope_options.keys()), format_func=lambda x: slope_options[x])
        
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        
        thal_options = {3: "Normal", 6: "Fixed", 7: "Reversible"}
        thal = st.selectbox("Thalassemia", list(thal_options.keys()), format_func=lambda x: thal_options[x])

    submitted = st.form_submit_button("Predict Heart Disease")

if submitted:
    # --- Step 1: One-hot encoded features expected by model
    input_data = {
        'exang': exang,
        'ca': ca,
        'cp_3.0': 1 if cp == 3 else 0,
        'cp_4.0': 1 if cp == 4 else 0,
        'thal_7.0': 1 if thal == 7 else 0
    }

    input_df = pd.DataFrame([input_data])

    # --- Step 2: Load and predict using model
    try:
        model = joblib.load(model_path)
        prediction = model.predict(input_df)[0]

        # --- Step 3: Display result
        st.subheader("🎯 Prediction Result")
        if prediction == 1:
            st.error("🚨 **HIGH RISK**: Heart disease detected!")
            st.warning("Please consult with a healthcare professional immediately.")
        else:
            st.success("✅ **LOW RISK**: No heart disease detected.")
            st.info("Continue maintaining a healthy lifestyle.")
        

        
        # Show all input values for reference
        st.subheader("📊 Complete Patient Data")
        complete_data = {
            'Age': age,
            'Sex': "Female" if sex == 0 else "Male",
            'Chest Pain Type': cp_options[cp],
            'Resting BP': f"{trestbps} mm Hg",
            'Cholesterol': f"{chol} mg/dl",
            'Fasting Blood Sugar > 120': "Yes" if fbs == 1 else "No",
            'Resting ECG': restecg_options[restecg],
            'Max Heart Rate': thalach,
            'Exercise Angina': "Yes" if exang == 1 else "No",
            'ST Depression': oldpeak,
            'Slope': slope_options[slope],
            'Major Vessels': ca,
            'Thalassemia': thal_options[thal]
        }
        st.json(complete_data)

    except Exception as e:
        st.error(f"❌ Model loading or prediction failed:")
        st.code(str(e))
        st.info("Please check if the model file is corrupted or missing required dependencies.") 