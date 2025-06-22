import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Healthcare Trends",
    page_icon="📈",
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
st.markdown('<h1 class="beautiful-title">📈 Healthcare Trends Analysis</h1>', unsafe_allow_html=True)

st.markdown("""
This page provides comprehensive insights into heart disease trends and patterns from the Cleveland Heart Disease dataset.
""")

# Load the training data
try:
    data_path = "data/Trained_cleaned_heart_data.csv"
    df = pd.read_csv(data_path)
    
    # Check if required columns exist
    required_columns = ['sex', 'target', 'exang', 'fbs', 'cp_2.0', 'cp_3.0', 'cp_4.0', 'restecg_1.0', 'restecg_2.0']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        st.info("Please ensure the training data file has the correct structure.")
        st.stop()
    
    st.success("✅ Data loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("Please ensure the data file exists in the correct location.")
    st.stop()

# Display basic statistics
st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>👥 Total Patients</h3>
        <h2>{len(df):,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    heart_disease_count = df['target'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <h3>❤️ Heart Disease Cases</h3>
        <h2>{heart_disease_count:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    disease_rate = (heart_disease_count / len(df)) * 100
    st.markdown(f"""
    <div class="metric-card">
        <h3>📈 Disease Rate</h3>
        <h2>{disease_rate:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    healthy_count = len(df) - heart_disease_count
    st.markdown(f"""
    <div class="metric-card">
        <h3>✅ Healthy Patients</h3>
        <h2>{healthy_count:,}</h2>
    </div>
    """, unsafe_allow_html=True)

# Gender distribution
st.subheader("👥 Gender Distribution")
col1, col2 = st.columns(2)

with col1:
    gender_counts = df['sex'].value_counts()
    gender_labels = ['Female', 'Male']
    gender_values = [gender_counts.get(0.0, 0), gender_counts.get(1.0, 0)]
    
    fig_gender = px.pie(
        values=gender_values,
        names=gender_labels,
        title="Gender Distribution",
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    fig_gender.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_gender, use_container_width=True)

with col2:
    # Gender vs Heart Disease
    gender_disease = df.groupby('sex')['target'].agg(['count', 'sum']).reset_index()
    gender_disease['disease_rate'] = (gender_disease['sum'] / gender_disease['count']) * 100
    gender_disease['sex_label'] = gender_disease['sex'].map({0.0: 'Female', 1.0: 'Male'})
    
    fig_gender_disease = px.bar(
        gender_disease,
        x='sex_label',
        y='disease_rate',
        title="Heart Disease Rate by Gender",
        color='sex_label',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    fig_gender_disease.update_layout(yaxis_title="Disease Rate (%)")
    st.plotly_chart(fig_gender_disease, use_container_width=True)

# Age distribution
st.subheader("📅 Age Distribution Analysis")
col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x='age',
        nbins=20,
        title="Age Distribution",
        color_discrete_sequence=['#2E8B57']
    )
    fig_age.update_layout(xaxis_title="Age", yaxis_title="Count")
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    # Age groups - handle normalized values (0-1 range)
    if df['age'].max() <= 1.0:  # Normalized data
        df['age_group'] = pd.cut(df['age'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                labels=['<20%', '20-40%', '40-60%', '60-80%', '80%+'])
    else:  # Raw age data
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 100], 
                                labels=['<40', '40-50', '50-60', '60-70', '70+'])
    
    age_group_disease = df.groupby('age_group')['target'].agg(['count', 'sum']).reset_index()
    age_group_disease['disease_rate'] = (age_group_disease['sum'] / age_group_disease['count']) * 100
    
    fig_age_disease = px.bar(
        age_group_disease,
        x='age_group',
        y='disease_rate',
        title="Heart Disease Rate by Age Group",
        color='disease_rate',
        color_continuous_scale='Reds'
    )
    fig_age_disease.update_layout(yaxis_title="Disease Rate (%)")
    st.plotly_chart(fig_age_disease, use_container_width=True)

# Risk factors analysis
st.subheader("⚠️ Risk Factors Analysis")

# Create subplots for multiple risk factors
fig_risk = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Chest Pain Type', 'Exercise Angina', 'Blood Sugar > 120', 'ECG Results'),
    specs=[[{"type": "pie"}, {"type": "pie"}],
           [{"type": "pie"}, {"type": "pie"}]]
)

# Chest Pain Type - using one-hot encoded columns
cp_2_count = df['cp_2.0'].sum()
cp_3_count = df['cp_3.0'].sum()
cp_4_count = df['cp_4.0'].sum()
cp_1_count = len(df) - cp_2_count - cp_3_count - cp_4_count

cp_counts = [cp_1_count, cp_2_count, cp_3_count, cp_4_count]
cp_labels = ['Typical Angina', 'Atypical Angina', 'Non-anginal', 'Asymptomatic']
fig_risk.add_trace(
    go.Pie(labels=cp_labels, values=cp_counts, name="Chest Pain"),
    row=1, col=1
)

# Exercise Angina
exang_counts = df['exang'].value_counts()
exang_labels = ['No', 'Yes']
fig_risk.add_trace(
    go.Pie(labels=exang_labels, values=exang_counts.values, name="Exercise Angina"),
    row=1, col=2
)

# Blood Sugar
fbs_counts = df['fbs'].value_counts()
fbs_labels = ['≤ 120', '> 120']
fig_risk.add_trace(
    go.Pie(labels=fbs_labels, values=fbs_counts.values, name="Blood Sugar"),
    row=2, col=1
)

# ECG Results - using one-hot encoded columns
restecg_1_count = df['restecg_1.0'].sum()
restecg_2_count = df['restecg_2.0'].sum()
restecg_0_count = len(df) - restecg_1_count - restecg_2_count

restecg_counts = [restecg_0_count, restecg_1_count, restecg_2_count]
restecg_labels = ['Normal', 'ST-T Abnormality', 'LV Hypertrophy']
fig_risk.add_trace(
    go.Pie(labels=restecg_labels, values=restecg_counts, name="ECG Results"),
    row=2, col=2
)

fig_risk.update_layout(height=600, title_text="Risk Factors Distribution")
st.plotly_chart(fig_risk, use_container_width=True)

# Key insights
st.subheader("🔍 Key Insights")

insights = [
    f"• **Gender Impact**: Males have a {gender_disease[gender_disease['sex']==1.0]['disease_rate'].iloc[0]:.1f}% heart disease rate vs {gender_disease[gender_disease['sex']==0.0]['disease_rate'].iloc[0]:.1f}% for females",
    f"• **Age Risk**: The highest disease rate is in the {age_group_disease.loc[age_group_disease['disease_rate'].idxmax(), 'age_group']} age group",
    f"• **Exercise Angina**: {exang_counts.get(1.0, 0)} patients ({exang_counts.get(1.0, 0)/len(df)*100:.1f}%) experience exercise-induced angina",
    f"• **High Blood Sugar**: {fbs_counts.get(1.0, 0)} patients ({fbs_counts.get(1.0, 0)/len(df)*100:.1f}%) have fasting blood sugar > 120 mg/dl",
    f"• **Chest Pain**: {cp_4_count} patients ({cp_4_count/len(df)*100:.1f}%) are asymptomatic",
    f"• **ECG Abnormalities**: {restecg_1_count + restecg_2_count} patients have ECG abnormalities"
]

for insight in insights:
    st.markdown(insight)

# Correlation heatmap
st.subheader("📊 Feature Correlations")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

fig_corr = px.imshow(
    correlation_matrix,
    title="Feature Correlation Heatmap",
    color_continuous_scale='RdBu',
    aspect='auto'
)
fig_corr.update_layout(height=600)
st.plotly_chart(fig_corr, use_container_width=True) 