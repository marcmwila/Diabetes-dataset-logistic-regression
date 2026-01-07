# diabetes_logistic_regression.py
"""
Diabetes Prediction using Logistic Regression
Author: Marc Mwila
Project: Portfolio Data Science Project
"""

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import pickle
import os

# 2️⃣ Load Dataset
DATA_PATH = "data/diabetes.csv"  # Make sure this path is correct
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Please download the dataset from Kaggle and place it in the 'data/' folder.")

df = pd.read_csv(DATA_PATH)
print("\nDataset Loaded Successfully!")
print(df.head())

# 3️⃣ Basic EDA
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nClass Distribution:")
print(df['Outcome'].value_counts())

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Countplot for Outcome
sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Outcome Distribution")
plt.show()

# Histograms of Features
df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()

# 4️⃣ Data Preprocessing
# Replace zero values with median in specific columns
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df[col] = df[col].replace(0, df[col].median())

# Split Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5️⃣ Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
print("\nModel Trained Successfully!")

# 6️⃣ Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='blue')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 7️⃣ Save Model
MODEL_PATH = "models/logistic_model.pkl"
os.makedirs("models", exist_ok=True)

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"\nTrained Model Saved Successfully at {MODEL_PATH}!")

# 8️⃣ Predict on New Data (Optional Example)
example_patient = np.array([[2, 120, 70, 30, 100, 25.0, 0.5, 30]])  # example features
example_patient_scaled = scaler.transform(example_patient)
prediction = model.predict(example_patient_scaled)
print(f"\nPrediction for Example Patient: {'Diabetic' if prediction[0]==1 else 'Not Diabetic'}")
