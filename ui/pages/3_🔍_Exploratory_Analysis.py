import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="🔍",
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
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        text-align: center;
        margin: 10px 0;
        color: #333;
    }
    .metric-card h3 {
        color: #2E8B57;
        margin-bottom: 10px;
    }
    .metric-card h2 {
        color: #333;
        font-size: 2rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Beautiful title
st.markdown('<h1 class="beautiful-title">🔍 Exploratory Data Analysis</h1>', unsafe_allow_html=True)

st.markdown("""
This page provides a comprehensive exploratory analysis of the original Cleveland Heart Disease dataset.
""")

# Load the original data
try:
    data_path = "data/processed.cleveland.data"
    # Define column names based on the Cleveland dataset
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
               'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.read_csv(data_path, names=columns, na_values='?')
    
    # Clean the data - remove rows with missing values
    df_clean = df.dropna()
    
    if len(df_clean) == 0:
        st.error("❌ No valid data found after cleaning missing values")
        st.stop()
    
    st.success(f"✅ Original Cleveland dataset loaded successfully! ({len(df_clean)} valid records)")
    df = df_clean  # Use cleaned data
    
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("Please ensure the original data file exists in the correct location.")
    st.stop()

# Data overview
st.subheader("📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 Total Records</h3>
        <h2>{len(df):,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    missing_count = df.isnull().sum().sum()
    st.markdown(f"""
    <div class="metric-card">
        <h3>❓ Missing Values</h3>
        <h2>{missing_count:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    heart_disease_count = df['target'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <h3>❤️ Heart Disease</h3>
        <h2>{heart_disease_count:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    disease_rate = (heart_disease_count / len(df)) * 100
    st.markdown(f"""
    <div class="metric-card">
        <h3>📈 Disease Rate</h3>
        <h2>{disease_rate:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

# Data info
st.subheader("📊 Data Information")
col1, col2 = st.columns(2)

with col1:
    st.write("**Dataset Shape:**", df.shape)
    st.write("**Features:**", len(df.columns) - 1)  # Excluding target
    st.write("**Target Variable:**", "Heart Disease (0/1)")

with col2:
    st.write("**Data Types:**")
    st.write(df.dtypes.value_counts())

# Missing values analysis
if missing_count > 0:
    st.subheader("❓ Missing Values Analysis")
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percent.values
    }).sort_values('Missing Count', ascending=False)
    
    fig_missing = px.bar(
        missing_df[missing_df['Missing Count'] > 0],
        x='Column',
        y='Missing Count',
        title="Missing Values by Column",
        color='Missing Percentage',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_missing, use_container_width=True)

# Feature distributions
st.subheader("📈 Feature Distributions")

# Select features to analyze
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Numeric features
st.write("**Numeric Features Distribution:**")
fig_numeric = make_subplots(
    rows=2, cols=3,
    subplot_titles=numeric_features,
    specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
           [{"type": "histogram"}, {"type": "histogram"}, {"type": "scatter"}]]
)

for i, feature in enumerate(numeric_features):
    row = (i // 3) + 1
    col = (i % 3) + 1
    
    if feature == 'oldpeak':
        # Scatter plot for oldpeak vs target
        fig_numeric.add_trace(
            go.Scatter(x=df[feature], y=df['target'], mode='markers', name=feature),
            row=row, col=col
        )
    else:
        fig_numeric.add_trace(
            go.Histogram(x=df[feature], name=feature),
            row=row, col=col
        )

fig_numeric.update_layout(height=600, title_text="Numeric Features Distribution")
st.plotly_chart(fig_numeric, use_container_width=True)

# Categorical features
st.write("**Categorical Features Distribution:**")
fig_categorical = make_subplots(
    rows=3, cols=3,
    subplot_titles=categorical_features,
    specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}],
           [{"type": "pie"}, {"type": "pie"}, {"type": "pie"}],
           [{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]]
)

# Define labels for categorical features
categorical_labels = {
    'sex': ['Female', 'Male'],
    'cp': ['Typical Angina', 'Atypical Angina', 'Non-anginal', 'Asymptomatic'],
    'fbs': ['≤ 120', '> 120'],
    'restecg': ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'],
    'exang': ['No', 'Yes'],
    'slope': ['Upsloping', 'Flat', 'Downsloping'],
    'ca': ['0', '1', '2', '3'],
    'thal': ['Normal', 'Fixed', 'Reversible']
}

for i, feature in enumerate(categorical_features):
    row = (i // 3) + 1
    col = (i % 3) + 1
    
    counts = df[feature].value_counts()
    labels = categorical_labels.get(feature, [str(x) for x in counts.index])
    
    fig_categorical.add_trace(
        go.Pie(labels=labels, values=counts.values, name=feature),
        row=row, col=col
    )

fig_categorical.update_layout(height=800, title_text="Categorical Features Distribution")
st.plotly_chart(fig_categorical, use_container_width=True)

# Target distribution
st.subheader("🎯 Target Variable Analysis")

col1, col2 = st.columns(2)

with col1:
    target_counts = df['target'].value_counts()
    
    # Handle different target values (0, 1, 2, 3, 4)
    unique_targets = sorted(target_counts.index)
    if len(unique_targets) == 2:
        target_labels = ['No Heart Disease', 'Heart Disease']
    else:
        target_labels = [f'Disease Level {t}' for t in unique_targets]
    
    # Ensure we have matching lengths
    if len(target_counts) == len(target_labels):
        fig_target = px.pie(
            values=target_counts.values,
            names=target_labels,
            title="Target Variable Distribution",
            color_discrete_sequence=['#4ECDC4', '#FF6B6B']
        )
        fig_target.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_target, use_container_width=True)
    else:
        st.write(f"Target variable has {len(target_counts)} categories: {list(target_counts.index)}")

with col2:
    # Target by gender
    target_gender = df.groupby('sex')['target'].agg(['count', 'sum']).reset_index()
    target_gender['disease_rate'] = (target_gender['sum'] / target_gender['count']) * 100
    target_gender['sex_label'] = target_gender['sex'].map({0: 'Female', 1: 'Male'})
    
    fig_target_gender = px.bar(
        target_gender,
        x='sex_label',
        y='disease_rate',
        title="Heart Disease Rate by Gender",
        color='sex_label',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    fig_target_gender.update_layout(yaxis_title="Disease Rate (%)")
    st.plotly_chart(fig_target_gender, use_container_width=True)

# Correlation analysis
st.subheader("📊 Correlation Analysis")

# Calculate correlation matrix
numeric_df = df[numeric_features + ['target']].dropna()
correlation_matrix = numeric_df.corr()

fig_corr = px.imshow(
    correlation_matrix,
    title="Feature Correlation Heatmap",
    color_continuous_scale='RdBu',
    aspect='auto'
)
fig_corr.update_layout(height=600)
st.plotly_chart(fig_corr, use_container_width=True)

# Statistical summary
st.subheader("📋 Statistical Summary")
st.write(df.describe())

# Data quality insights
st.subheader("🔍 Data Quality Insights")

insights = [
    f"• **Dataset Size**: {len(df)} records with {len(df.columns)} features",
    f"• **Missing Values**: {missing_count} total missing values across all features",
    f"• **Heart Disease Rate**: {disease_rate:.1f}% of patients have heart disease",
    f"• **Age Range**: {df['age'].min()} - {df['age'].max()} years",
    f"• **Gender Distribution**: {df['sex'].value_counts().get(0, 0)} females, {df['sex'].value_counts().get(1, 0)} males",
    f"• **Chest Pain Types**: {len(df['cp'].unique())} different types of chest pain",
    f"• **Blood Pressure Range**: {df['trestbps'].min()} - {df['trestbps'].max()} mm Hg",
    f"• **Cholesterol Range**: {df['chol'].min()} - {df['chol'].max()} mg/dl"
]

for insight in insights:
    st.markdown(insight)

# Recommendations
st.subheader("💡 Data Preprocessing Recommendations")

recommendations = [
    "• **Handle Missing Values**: Consider imputation or removal based on missing value patterns",
    "• **Feature Engineering**: Create age groups and other derived features",
    "• **Outlier Detection**: Check for outliers in numeric features like cholesterol and blood pressure",
    "• **Feature Scaling**: Apply normalization/standardization for machine learning models",
    "• **Categorical Encoding**: Convert categorical variables to numerical format",
    "• **Data Validation**: Ensure all values are within expected ranges"
]

for recommendation in recommendations:
    st.markdown(recommendation) 