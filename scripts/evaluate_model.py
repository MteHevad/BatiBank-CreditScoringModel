import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Load the test dataset
data = pd.read_csv('cleaned_data.csv')

# Load the model
model = joblib.load('credit_risk_model.pkl')

# Split data (assuming FraudResult is your target)
X_test = data.drop(['FraudResult'], axis=1)
y_test = data['FraudResult']

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
