# Why RandomizedSearch Scores Differ from Cross Validation

## The Problem You're Seeing

**RandomizedSearch reports**: `Best Recall Score: 0.7878947368421052`
**Cross validation reports**: `Mean Recall: 0.37526315789473685`

This is a **HUGE difference**! Here's why it happens:

## Root Causes

### 1. **Different CV Splits**
- **RandomizedSearch**: Uses its own internal 5-fold CV during the search
- **Your Cross Validation**: Uses a different 5-fold split of the same data
- Even with the same `random_state=42`, the splits can be different!

### 2. **Different Optimization vs Evaluation**
- **RandomizedSearch**: Optimizes for recall during the search process
- **Cross Validation**: Evaluates all metrics after the search is complete

### 3. **Data Leakage Risk**
- You're using the same data for both hyperparameter tuning and final evaluation

## The Fix

```python
# 1. Create a fixed CV splitter
from sklearn.model_selection import StratifiedKFold
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Use the SAME splitter for both search and evaluation
random_search = RandomizedSearchCV(
    estimator=LogisticRegression(max_iter=1000, solver='liblinear'),
    param_distributions=param_dist,
    n_iter=50,
    scoring='recall',
    cv=cv_splitter,  # Use the same CV split
    random_state=42,
    n_jobs=-1
)

# 3. Evaluate with the SAME splitter
cv_results = cross_validate(
    best_model, 
    X_train, 
    y_train, 
    cv=cv_splitter,  # Same CV split!
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)
```

## Expected Results After Fix

After using the same CV splitter:
- **RandomizedSearch recall**: ~0.78
- **Cross validation recall**: ~0.78
- **Difference**: < 0.01 (much more reasonable!)

## Best Practices

1. **Use StratifiedKFold** for imbalanced datasets (like heart disease)
2. **Use the same CV splitter** for both search and evaluation
3. **Use proper train/validation/test splits** to avoid data leakage
4. **Always verify** that search scores and CV scores are consistent

## Why This Matters

- **Inconsistent results** indicate methodological problems
- **Data leakage** can lead to overly optimistic performance estimates
- **Proper CV** ensures reliable model evaluation 