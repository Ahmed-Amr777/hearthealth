# Making Your Heart Disease Model More Complex

## Why Increase Model Complexity?

Your current logistic regression model might be **underfitting** the data. More complex models can capture:
- **Non-linear relationships** between features
- **Feature interactions** 
- **Complex decision boundaries**
- **Hidden patterns** in the data

## 1. Feature Engineering (Low Complexity Increase)

### Polynomial Features
```python
from sklearn.preprocessing import PolynomialFeatures

# Create quadratic features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Now use with any model
model.fit(X_poly, y)
```

### Interaction Features
```python
# Create interaction between important features
X['age_chol_interaction'] = X['age'] * X['chol']
X['age_trestbps_interaction'] = X['age'] * X['trestbps']
```

### Domain-Specific Features
```python
# Heart rate zones
X['hr_zone'] = pd.cut(X['thalach'], bins=[0, 120, 140, 160, 300], labels=[1,2,3,4])

# Age groups
X['age_group'] = pd.cut(X['age'], bins=[0, 45, 55, 65, 100], labels=[1,2,3,4])
```

## 2. Ensemble Methods (Medium Complexity)

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Tree depth
    min_samples_split=5,   # Minimum samples to split
    random_state=42
)
```

### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,     # Learning rate
    max_depth=3,           # Tree depth
    random_state=42
)
```

### Voting Classifier
```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ],
    voting='soft'  # Use probability predictions
)
```

## 3. Support Vector Machines (Medium-High Complexity)

### Different Kernels
```python
from sklearn.svm import SVC

# Linear kernel
svm_linear = SVC(kernel='linear', probability=True)

# RBF kernel (non-linear)
svm_rbf = SVC(kernel='rbf', probability=True)

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, probability=True)
```

## 4. Neural Networks (High Complexity)

### Simple Neural Network
```python
from sklearn.neural_network import MLPClassifier

mlp_simple = MLPClassifier(
    hidden_layer_sizes=(50, 25),  # Two hidden layers
    max_iter=1000,
    random_state=42
)
```

### Complex Neural Network
```python
mlp_complex = MLPClassifier(
    hidden_layer_sizes=(100, 50, 25, 10),  # Four hidden layers
    max_iter=1000,
    alpha=0.01,  # L2 regularization
    random_state=42
)
```

### Deep Learning (TensorFlow/Keras)
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

## 5. Advanced Techniques

### Stacking
```python
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier()),
    ('gb', GradientBoostingClassifier()),
    ('svm', SVC(probability=True))
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
```

### Feature Selection + Complex Model
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select best features
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Use with complex model
rf = RandomForestClassifier()
rf.fit(X_selected, y)
```

## 6. Hyperparameter Tuning for Complex Models

### Random Forest Tuning
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

rf_tuned = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='roc_auc'
)
```

### Neural Network Tuning
```python
param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

mlp_tuned = RandomizedSearchCV(
    MLPClassifier(max_iter=1000),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='roc_auc'
)
```

## 7. Evaluation Strategy

### Cross Validation with Multiple Metrics
```python
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

results = cross_validate(
    model, X, y, cv=5,
    scoring=scoring,
    return_train_score=True
)
```

### Overfitting Detection
```python
# Compare train vs test scores
for metric in scoring:
    train_score = results[f'train_{metric}'].mean()
    test_score = results[f'test_{metric}'].mean()
    diff = train_score - test_score
    print(f"{metric}: Train-Test Diff = {diff:.4f}")
```

## 8. Recommended Progression

### Step 1: Start Simple
```python
# Polynomial features + Logistic Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
lr = LogisticRegression()
lr.fit(X_poly, y)
```

### Step 2: Add Ensemble
```python
# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
```

### Step 3: Try Neural Network
```python
# MLP with engineered features
mlp = MLPClassifier(hidden_layer_sizes=(100, 50))
mlp.fit(X_poly, y)
```

### Step 4: Optimize Best Model
```python
# Hyperparameter tuning
best_model = RandomizedSearchCV(
    best_estimator,
    param_distributions,
    n_iter=100,
    cv=5
)
```

## 9. Performance Expectations

| Model Complexity | Expected ROC-AUC | Training Time | Interpretability |
|------------------|------------------|---------------|------------------|
| Logistic Regression | 0.75-0.80 | Fast | High |
| Random Forest | 0.80-0.85 | Medium | Medium |
| Gradient Boosting | 0.82-0.87 | Medium | Medium |
| Neural Network | 0.83-0.88 | Slow | Low |
| Deep Learning | 0.85-0.90 | Very Slow | Very Low |

## 10. Trade-offs to Consider

### Pros of Complex Models:
- **Better performance** on complex patterns
- **Captures non-linear relationships**
- **Can handle feature interactions**

### Cons of Complex Models:
- **Overfitting risk** (especially with small datasets)
- **Longer training time**
- **Less interpretable**
- **More hyperparameters to tune**

## 11. Best Practices

1. **Start simple** and gradually increase complexity
2. **Use cross-validation** to prevent overfitting
3. **Monitor train vs test scores**
4. **Consider interpretability** requirements
5. **Balance performance vs complexity**
6. **Use proper train/validation/test splits**

## 12. For Your Heart Disease Dataset

Given your dataset size (~300 samples), I recommend:

1. **Random Forest** (good balance of performance and interpretability)
2. **Gradient Boosting** (if you want better performance)
3. **Polynomial features + Logistic Regression** (if interpretability is crucial)
4. **Simple Neural Network** (if you want to try deep learning)

**Avoid**: Very complex deep learning models (risk of overfitting with small dataset) 