import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")


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
# 3. One-Hot Encode Transaction Type
# -----------------------------------

df_encoded = pd.get_dummies(
    df_model,
    columns=["type"],
    drop_first=True,
    dtype=int
)


# -----------------------------------
# 4. Separate Features and Target
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
# 6. Apply SMOTE to Training Data Only
# -----------------------------------

smote = SMOTE(
    sampling_strategy=0.1,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


# -----------------------------------
# 7. Train Random Forest
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train_smote, y_train_smote)

print("Training completed successfully! ✅")


# -----------------------------------
# 8. Predictions
# -----------------------------------

print("\nMaking predictions...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# -----------------------------------
# 9. Evaluation
# -----------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))


# -----------------------------------
# 10. Save Model
# -----------------------------------

joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

print("\nRandom Forest model saved successfully! ✅")
