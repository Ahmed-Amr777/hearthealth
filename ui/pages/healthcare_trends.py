import streamlit as st
import pandas as pd

def show_healthcare_trends():
    """Healthcare Trends Page"""
    st.title("📈 Healthcare Trends")
    
    # Load the processed Cleveland data
    try:
        # Define column names for the Cleveland dataset
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        # Load the processed Cleveland data
        data = pd.read_csv("data/processed.cleveland.data", names=columns, na_values='?')
        
        # Calculate actual statistics from the data
        total_patients = len(data)
        heart_disease_cases = data['target'].sum()
        no_disease_cases = total_patients - heart_disease_cases
        disease_rate = (heart_disease_cases / total_patients) * 100
        
        # Age statistics
        age_stats = data['age'].describe()
        avg_age = age_stats['mean']
        
        # Gender distribution
        male_count = data['sex'].sum()
        female_count = total_patients - male_count
        male_percentage = (male_count / total_patients) * 100
        
        # Risk factor analysis
        exang_cases = data['exang'].sum()
        exang_rate = (exang_cases / total_patients) * 100
        
        # Chest pain analysis
        cp_counts = data['cp'].value_counts()
        cp_1_rate = (cp_counts.get(1, 0) / total_patients) * 100
        cp_2_rate = (cp_counts.get(2, 0) / total_patients) * 100
        cp_3_rate = (cp_counts.get(3, 0) / total_patients) * 100
        cp_4_rate = (cp_counts.get(4, 0) / total_patients) * 100
        
        # Thalassemia analysis
        thal_counts = data['thal'].value_counts()
        thal_3_rate = (thal_counts.get(3, 0) / total_patients) * 100
        thal_6_rate = (thal_counts.get(6, 0) / total_patients) * 100
        thal_7_rate = (thal_counts.get(7, 0) / total_patients) * 100
        
        # Blood pressure analysis
        bp_stats = data['trestbps'].describe()
        high_bp_cases = len(data[data['trestbps'] > 140])  # High BP threshold
        high_bp_rate = (high_bp_cases / total_patients) * 100
        
        # Cholesterol analysis
        chol_stats = data['chol'].describe()
        high_chol_cases = len(data[data['chol'] > 200])  # High cholesterol threshold
        high_chol_rate = (high_chol_cases / total_patients) * 100
        
        st.subheader("📊 Dataset Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patients", f"{total_patients}")
            st.metric("Heart Disease Cases", f"{heart_disease_cases}")
            st.metric("Disease Rate", f"{disease_rate:.1f}%")
        
        with col2:
            st.metric("Average Age", f"{avg_age:.0f} years")
            st.metric("Male Patients", f"{male_count} ({male_percentage:.1f}%)")
            st.metric("Female Patients", f"{female_count} ({100-male_percentage:.1f}%)")
        
        with col3:
            st.metric("Exercise Angina", f"{exang_cases} ({exang_rate:.1f}%)")
            st.metric("High Blood Pressure", f"{high_bp_cases} ({high_bp_rate:.1f}%)")
            st.metric("High Cholesterol", f"{high_chol_cases} ({high_chol_rate:.1f}%)")
        
        st.subheader("🎯 Key Risk Factors Analysis")
        
        # Display risk factors
        risk_factors = {
            'Exercise Induced Angina': exang_rate,
            'High Blood Pressure (>140)': high_bp_rate,
            'High Cholesterol (>200)': high_chol_rate,
            'Chest Pain Type 1 (Typical)': cp_1_rate,
            'Chest Pain Type 2 (Atypical)': cp_2_rate,
            'Chest Pain Type 3 (Non-anginal)': cp_3_rate,
            'Chest Pain Type 4 (Asymptomatic)': cp_4_rate,
            'Thalassemia Normal': thal_3_rate,
            'Thalassemia Fixed Defect': thal_6_rate,
            'Thalassemia Reversible': thal_7_rate
        }
        
        for factor, percentage in risk_factors.items():
            st.write(f"• **{factor}**: {percentage:.1f}% of patients")
        
        st.subheader("📈 Data Distribution")
        
        # Show target distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Heart Disease Status:**")
            st.write(f"✅ No Disease: {no_disease_cases} patients ({100-disease_rate:.1f}%)")
            st.write(f"🚨 Heart Disease: {heart_disease_cases} patients ({disease_rate:.1f}%)")
        
        with col2:
            st.write("**Gender Distribution:**")
            st.write(f"👨 Male: {male_count} patients ({male_percentage:.1f}%)")
            st.write(f"👩 Female: {female_count} patients ({100-male_percentage:.1f}%)")
        
        st.subheader("🔬 Feature Importance")
        
        st.write("Based on our SVM model analysis, the most important features are:")
        important_features = [
            "Exercise Induced Angina (exang)",
            "Number of Major Vessels (ca)", 
            "Chest Pain Type 3 (cp_3.0)",
            "Chest Pain Type 4 (cp_4.0)",
            "Thalassemia 7 (thal_7.0)"
        ]
        
        for i, feature in enumerate(important_features, 1):
            st.write(f"{i}. **{feature}**")
        
        st.subheader("💡 Insights")
        
        insights = [
            f"• **{disease_rate:.1f}%** of patients in our dataset have heart disease",
            f"• **{exang_rate:.1f}%** experience exercise-induced angina",
            f"• **{high_bp_rate:.1f}%** have high blood pressure (>140 mmHg)",
            f"• **{high_chol_rate:.1f}%** have high cholesterol (>200 mg/dl)",
            f"• **{cp_3_rate:.1f}%** have non-anginal chest pain",
            f"• **{cp_4_rate:.1f}%** have asymptomatic chest pain",
            f"• **{thal_7_rate:.1f}%** have reversible thalassemia defect",
            f"• Average patient age is **{avg_age:.0f} years**",
            f"• **{male_percentage:.1f}%** of patients are male"
        ]
        
        for insight in insights:
            st.write(insight)
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the 'processed.cleveland.data' file is available in the data/ directory.") 