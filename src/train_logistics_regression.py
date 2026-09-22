import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# -----------------------------------
# 1. Load Dataset
# -----------------------------------

file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

# -----------------------------------
# 2. Select Features
# -----------------------------------

features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

target = "isFraud"

df_model = df[features + [target]].copy()

# -----------------------------------
# 3. Encode Transaction Type
# -----------------------------------

df_encoded = pd.get_dummies(
    df_model,
    columns=["type"],
    drop_first=True,
    dtype=int
)

# -----------------------------------
# 4. Separate X and y
# -----------------------------------

X = df_encoded.drop("isFraud", axis=1)
y = df_encoded["isFraud"]

# -----------------------------------
# 5. Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------------
# 6. Feature Scaling
# -----------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------
# 7. Apply SMOTE ONLY to Training Data
# -----------------------------------

smote = SMOTE(
    sampling_strategy=0.1,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("After SMOTE:")
print(y_train_smote.value_counts())

# -----------------------------------
# 8. Train Logistic Regression
# -----------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print("\nTraining Logistic Regression...")

model.fit(X_train_smote, y_train_smote)

print("Training completed successfully! ✅")

# -----------------------------------
# 9. Make Predictions
# -----------------------------------

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("\nMaking predictions...")

y_pred = model.predict(X_test_scaled)

# Probability predictions for ROC-AUC
y_prob = model.predict_proba(X_test_scaled)[:, 1]


# -----------------------------------
# 10. Evaluation Metrics
# -----------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))



import joblib

# Save Logistic Regression model
joblib.dump(model, "models/logistic_regression_model.pkl")

# Save the scaler too
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel saved successfully! ✅")
print("Scaler saved successfully! ✅")