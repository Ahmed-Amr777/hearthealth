import streamlit as st
import pandas as pd
import joblib

def show_heart_disease_detection():
    """Heart Disease Detection Page"""
    st.title("Heart Disease Detection")

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

        submitted = st.form_submit_button("Predict")

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
            model = joblib.load("models/svm_heart_model.pkl")
            prediction = model.predict(input_df)[0]

            # --- Step 3: Display result
            st.subheader("Result")
            if prediction == 1:
                st.error("🚨 High risk of heart disease detected!")
            else:
                st.success("✅ No heart disease detected.")
            
            st.subheader("Input Summary")
            st.json(input_data)

        except Exception as e:
            st.error(f"Model loading or prediction failed:\n{e}") 