# Heart Disease Prediction Project

A comprehensive machine learning project for predicting heart disease using various algorithms and advanced preprocessing techniques.

## 📋 Project Overview

This project implements a complete machine learning pipeline for heart disease prediction using the Cleveland Heart Disease dataset. The project includes data preprocessing, feature engineering, multiple ML algorithms, and comprehensive evaluation.

## 🎯 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection and prediction can significantly improve patient outcomes. This project aims to build accurate machine learning models to predict the presence of heart disease based on various clinical parameters.

## 📊 Dataset

- **Source**: Cleveland Heart Disease Dataset
- **Features**: 13 clinical parameters including age, sex, chest pain type, blood pressure, cholesterol, etc.
- **Target**: Binary classification (0: No disease, 1: Disease present)
- **Size**: 303 samples (after cleaning)

### Features Included:
- `age`: Age in years
- `sex`: Gender (1 = male, 0 = female)
- `cp`: Chest pain type (1-4)
- `trestbps`: Resting blood pressure
- `chol`: Serum cholesterol
- `fbs`: Fasting blood sugar
- `restecg`: Resting electrocardiographic results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of peak exercise ST segment
- `ca`: Number of major vessels colored by fluoroscopy
- `thal`: Thalassemia

## 🏗️ Project Structure

```
project/
├── data/                          # Data files
│   ├── processed.cleveland.data   # Raw dataset
│   ├── Trained_cleaned_heart_data.csv
│   ├── Test_cleaned_heart_data.csv
│   ├── Trained_pca_heart.csv
│   ├── Test_pca_heart.csv
│   ├── Trained_selected_features.csv
│   └── Test_selected_features.csv
├── notebooks/                     # Jupyter notebooks
│   ├── data_preprocessing.ipynb   # Data cleaning and preprocessing
│   ├── PCA.ipynb                  # Principal Component Analysis
│   ├── featureSelection.ipynb     # Feature selection methods
│   ├── hyper_parameter_tuning.ipynb # Model optimization
│   ├── supervised.ipynb           # Supervised learning models
│   └── unsupervised.ipynb         # Clustering analysis
├── models/                        # Saved models
│   └── svm_heart_model.pkl        # Trained SVM model
├── deployment/                    # Model deployment files
├── ui/                           # User interface
│   └── app.py                    # Streamlit web app
├── run_app.py                    # App runner script
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required packages (see requirements.txt)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd heart-disease-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

## 🎯 Using the Trained Model

### Quick Start - Web Application

The easiest way to use your trained model is through the web interface:

```bash
# Run the app
python run_app.py
```

This will:
- Install required dependencies automatically
- Start a Streamlit web server
- Open the app in your browser at `http://localhost:8501`

### Features Used by the Model

Your trained SVM model uses **5 selected features**:
- `exang` (Exercise Induced Angina): 0 = No, 1 = Yes
- `ca` (Number of Major Vessels): 0-4 (normalized to 0-1)
- `cp_3.0` (Chest Pain Type 3): One-hot encoded
- `cp_4.0` (Chest Pain Type 4): One-hot encoded  
- `thal_7.0` (Thalassemia 7): One-hot encoded

### Programmatic Usage

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/svm_heart_model.pkl')

# Create input data with the 5 selected features
input_data = {
    'exang': 0,        # No exercise induced angina
    'ca': 0.0,         # 0 major vessels (normalized)
    'cp_3.0': 1,       # Chest pain type 3
    'cp_4.0': 0,       # Not chest pain type 4
    'thal_7.0': 0      # Not thalassemia 7
}

# Convert to DataFrame
df = pd.DataFrame([input_data])

# Make prediction
prediction = model.predict(df)[0]
probability = model.predict_proba(df)[0][1]

print(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
print(f"Probability: {probability:.2%}")
```

## 📈 Data Preprocessing Pipeline

The project implements a comprehensive preprocessing pipeline:

1. **Data Cleaning**
   - Handle missing values
   - Remove outliers using IQR method
   - Data type conversions

2. **Feature Engineering**
   - One-hot encoding for categorical variables
   - Log transformation for skewed features (chol)
   - Square root transformation for oldpeak

3. **Dimensionality Reduction**
   - Principal Component Analysis (PCA) - 3 components
   - Feature selection using Chi-square test

4. **Scaling**
   - MinMaxScaler for numerical features
   - StandardScaler for SVM

## 🤖 Machine Learning Models

### Supervised Learning Models
- **Logistic Regression**: Baseline model with regularization
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Advanced ensemble technique
- **XGBoost**: Optimized gradient boosting
- **Support Vector Machine (SVM)**: With different kernels (linear, RBF, poly)
- **Neural Networks**: Multi-layer perceptron

### Unsupervised Learning
- **K-Means Clustering**: Patient segmentation
- **Hierarchical Clustering**: Dendrogram analysis

### Model Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Cross-validation (5-fold stratified)

## 📊 Results

### Best Performing Models

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| XGBoost | 0.87 | 0.85 | 0.83 | 0.89 | 0.86 |
| Random Forest | 0.84 | 0.82 | 0.80 | 0.86 | 0.83 |
| SVM (RBF) | 0.83 | 0.81 | 0.79 | 0.85 | 0.82 |
| Logistic Regression | 0.78 | 0.76 | 0.74 | 0.80 | 0.77 |

### Key Findings
- **XGBoost** achieved the best performance with ROC-AUC of 0.87
- **Feature importance**: `ca` (number of vessels), `cp_4.0` (chest pain), `thal_7.0` (thalassemia) are most predictive
- **Data leakage prevention**: Proper train-test split and preprocessing pipeline
- **Cross-validation**: Consistent performance across folds

## 🔧 Usage

### Running the Complete Pipeline

1. **Data Preprocessing**
   ```python
   # Run data_preprocessing.ipynb
   # This creates cleaned datasets
   ```

2. **Feature Selection**
   ```python
   # Run featureSelection.ipynb
   # Identifies most important features
   ```

3. **Model Training**
   ```python
   # Run hyper_parameter_tuning.ipynb
   # Trains and optimizes models
   ```

4. **Evaluation**
   ```python
   # Results are automatically generated
   # Cross-validation scores and test set performance
   ```

### Quick Start Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load preprocessed data
df = pd.read_csv("data/Trained_selected_features.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(classification_report(y, predictions))
```

## 📁 Data Files

- `processed.cleveland.data`: Original dataset
- `Trained_cleaned_heart_data.csv`: Preprocessed training data
- `Test_cleaned_heart_data.csv`: Preprocessed test data
- `Trained_pca_heart.csv`: PCA-transformed training data
- `Test_pca_heart.csv`: PCA-transformed test data
- `Trained_selected_features.csv`: Final feature-selected training data
- `Test_selected_features.csv`: Final feature-selected test data

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development
- **XGBoost**: Gradient boosting implementation

## 🔍 Key Insights

1. **Feature Importance**: Number of major vessels (`ca`) is the most predictive feature
2. **Data Quality**: Proper preprocessing significantly improves model performance
3. **Model Selection**: Ensemble methods (XGBoost, Random Forest) outperform linear models
4. **Validation**: Cross-validation ensures robust performance estimates

## 🚀 Future Enhancements

- [ ] Web application for real-time predictions
- [ ] Additional datasets for model validation
- [ ] Deep learning models (CNN, RNN)
- [ ] Model interpretability tools (SHAP, LIME)
- [ ] Real-time data integration
- [ ] Mobile application

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Ahmed** - *Initial work* - [YourGitHub](https://github.com/Ahmed-Amr777)

## 🙏 Acknowledgments

- Cleveland Clinic Foundation for the dataset
- Scikit-learn community for excellent ML tools
- Open source community for various libraries

## 📞 Contact

- **Email**: ahmed.amr552255@gmail.com
- **GitHub**: [Your GitHub](https://github.com/Ahmed-Amr777)

---

**Note**: This project is for educational and research purposes. Medical decisions should not be based solely on this model's predictions.
