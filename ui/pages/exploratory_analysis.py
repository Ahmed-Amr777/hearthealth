import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_exploratory_analysis():
    """Exploratory Data Analysis Page"""
    st.title("🔍 Exploratory Data Analysis")
    
    # Load the original Cleveland data
    try:
        # Define column names for the Cleveland dataset
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        # Load the processed Cleveland data
        data = pd.read_csv("../data/processed.cleveland.data", names=columns, na_values='?')
        
        st.subheader("📋 Dataset Overview")
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Features", len(data.columns) - 1)  # Excluding target
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        # Data preview
        st.subheader("📊 Data Preview")
        st.dataframe(data.head(10))
        
        # Data info
        st.subheader("ℹ️ Data Information")
        buffer = st.empty()
        with st.spinner("Loading data information..."):
            info_buffer = []
            data.info(buf=info_buffer)
            buffer.text('\n'.join(info_buffer))
        
        # Missing values analysis
        st.subheader("🔍 Missing Values Analysis")
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data[missing_data > 0].plot(kind='bar', color='#ff6b6b')
            plt.title('Missing Values by Feature')
            plt.xlabel('Features')
            plt.ylabel('Missing Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.success("✅ No missing values found in the dataset!")
        
        # Target distribution
        st.subheader("🎯 Target Distribution")
        target_counts = data['target'].value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#4CAF50', '#f44336']
            plt.pie(target_counts.values, labels=['No Disease', 'Heart Disease'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('Heart Disease Distribution')
            st.pyplot(fig)
        
        with col2:
            st.write("**Target Statistics:**")
            st.write(f"• No Disease: {target_counts[0]} patients ({target_counts[0]/len(data)*100:.1f}%)")
            st.write(f"• Heart Disease: {target_counts[1]} patients ({target_counts[1]/len(data)*100:.1f}%)")
        
        # Age distribution
        st.subheader("👥 Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.hist(data['age'], bins=20, color='#4CAF50', alpha=0.7, edgecolor='black')
        plt.title('Age Distribution of Patients')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Age statistics
        age_stats = data['age'].describe()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Age", f"{age_stats['mean']:.1f}")
        with col2:
            st.metric("Median Age", f"{age_stats['50%']:.1f}")
        with col3:
            st.metric("Min Age", f"{age_stats['min']:.0f}")
        with col4:
            st.metric("Max Age", f"{age_stats['max']:.0f}")
        
        # Gender distribution
        st.subheader("🚻 Gender Distribution")
        gender_counts = data['sex'].value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#FF69B4', '#4169E1']
            plt.pie(gender_counts.values, labels=['Female', 'Male'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('Gender Distribution')
            st.pyplot(fig)
        
        with col2:
            st.write("**Gender Statistics:**")
            st.write(f"• Female: {gender_counts[0]} patients ({gender_counts[0]/len(data)*100:.1f}%)")
            st.write(f"• Male: {gender_counts[1]} patients ({gender_counts[1]/len(data)*100:.1f}%)")
        
        # Chest Pain Types
        st.subheader("💔 Chest Pain Types")
        cp_counts = data['cp'].value_counts().sort_index()
        cp_labels = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = plt.bar(range(len(cp_counts)), cp_counts.values, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        plt.title('Distribution of Chest Pain Types')
        plt.xlabel('Chest Pain Type')
        plt.ylabel('Count')
        plt.xticks(range(len(cp_counts)), cp_labels, rotation=45)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height}', ha='center', va='bottom')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        # Create correlation matrix for numerical features
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        corr_matrix = data[numerical_cols + ['target']].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Key insights
        st.subheader("💡 Key Insights from EDA")
        
        insights = [
            f"• **Dataset Size**: {len(data)} patients with {len(data.columns)-1} features",
            f"• **Age Range**: {age_stats['min']:.0f} to {age_stats['max']:.0f} years (mean: {age_stats['mean']:.1f})",
            f"• **Gender Balance**: {gender_counts[0]/len(data)*100:.1f}% female, {gender_counts[1]/len(data)*100:.1f}% male",
            f"• **Disease Prevalence**: {target_counts[1]/len(data)*100:.1f}% have heart disease",
            f"• **Most Common Chest Pain**: {cp_labels[cp_counts.idxmax()-1]} ({cp_counts.max()} patients)",
            f"• **Data Quality**: {'Good' if data.isnull().sum().sum() == 0 else 'Has missing values'}"
        ]
        
        for insight in insights:
            st.write(insight)
        
        # Feature descriptions
        st.subheader("📝 Feature Descriptions")
        
        feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male; 0 = female)',
            'cp': 'Chest pain type (1-4)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
            'restecg': 'Resting electrocardiographic results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1 = yes; 0 = no)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'Slope of peak exercise ST segment (1-3)',
            'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thal': 'Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)',
            'target': 'Heart disease (0 = no; 1 = yes)'
        }
        
        for feature, description in feature_descriptions.items():
            st.write(f"• **{feature}**: {description}")
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the 'processed.cleveland.data' file is available in the data/ directory.") 