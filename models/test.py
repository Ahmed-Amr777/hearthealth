import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
model = joblib.load("models/svm_heart_model.pkl")
print(model)
df = pd.read_csv("../data/test_selected_features.csv")
X_test = df.drop('target', axis=1)
y_test = df['target']
y_pred = model.predict(X_test)
print(y_pred)
print(y_test)
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
