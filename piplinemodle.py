import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
import warnings
warnings.filterwarnings('ignore')

# 1. DATA LOADING AND INITIAL PREPROCESSING
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv("data/processed.cleveland.data", names=columns, na_values='?')

# Remove rows with missing values
df = df[df['ca'].notna()]
df = df[df['thal'].notna()]

# 2. CATEGORICAL ENCODING
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True, dtype=int)

# 3. TARGET VARIABLE PROCESSING
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# 4. TRAIN-TEST SPLIT (AVOID DATA LEAKAGE)
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. FEATURE TRANSFORMATIONS (ONLY ON TRAINING DATA)
X_train_transformed = X_train.copy()
X_train_transformed['chol'] = np.log1p(X_train_transformed['chol'])
X_train_transformed['oldpeak'] = np.sqrt(X_train_transformed['oldpeak'])

X_test_transformed = X_test.copy()
X_test_transformed['chol'] = np.log1p(X_test_transformed['chol'])
X_test_transformed['oldpeak'] = np.sqrt(X_test_transformed['oldpeak'])

# 6. OUTLIER REMOVAL (ONLY ON TRAINING DATA)
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

outlier_columns = ['chol', 'trestbps', 'thalach', 'oldpeak']
X_train_clean = remove_outliers_iqr(X_train_transformed, outlier_columns)
y_train_clean = y_train.loc[X_train_clean.index]

# 7. SCALING
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

scaler = MinMaxScaler()
X_train_scaled = X_train_clean.copy()
X_train_scaled[numerical_features] = scaler.fit_transform(X_train_clean[numerical_features])

X_test_scaled = X_test_transformed.copy()
X_test_scaled[numerical_features] = scaler.transform(X_test_transformed[numerical_features])

# 8. PCA DIMENSION REDUCTION
pca_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled[pca_features])
X_test_pca = pca.transform(X_test_scaled[pca_features])

pca_train_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2', 'PC3'], index=X_train_scaled.index)
pca_test_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2', 'PC3'], index=X_test_scaled.index)

X_train_final = X_train_scaled.drop(columns=pca_features)
X_train_final = pd.concat([X_train_final, pca_train_df], axis=1)

X_test_final = X_test_scaled.drop(columns=pca_features)
X_test_final = pd.concat([X_test_final, pca_test_df], axis=1)

# 9. FEATURE SELECTION
Selected_features = ['exang', 'ca', 'cp_3.0', 'cp_4.0', 'thal_7.0']

X_train_selected = X_train_final[Selected_features]
y_train_selected = y_train_clean

X_test_selected = X_test_final[Selected_features]
y_test_selected = y_test

# 10. MODEL TRAINING
best_svm = SVC(
    C=0.19713872690045356,
    gamma=8.24431219090507,
    kernel='rbf',
    class_weight='balanced'
)

best_svm.fit(X_train_selected, y_train_selected)

# Fit the model
import joblib
joblib.dump(best_svm, 'models/pipline_svm_heart_model.pkl')
