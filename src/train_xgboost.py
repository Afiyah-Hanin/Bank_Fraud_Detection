import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib


# =====================================
# 1. LOAD DATASET
# =====================================

file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

print("Loading dataset...")
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")


# =====================================
# 2. SELECT FEATURES
# =====================================

selected_columns = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud"
]

df = df[selected_columns]


# =====================================
# 3. ONE-HOT ENCODING
# =====================================

df = pd.get_dummies(
    df,
    columns=["type"],
    drop_first=True,
    dtype=int
)


# =====================================
# 4. FEATURES AND TARGET
# =====================================

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# =====================================
# 5. TRAIN-TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# =====================================
# 6. HANDLE CLASS IMBALANCE WITH SMOTE
# =====================================

print("\nApplying SMOTE...")

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


# =====================================
# 7. TRAIN XGBOOST
# =====================================

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train_smote,
    y_train_smote
)

print("Training completed successfully! ✅")


# =====================================
# 8. MAKE PREDICTIONS
# =====================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# =====================================
# 9. EVALUATE MODEL
# =====================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

auc_score = roc_auc_score(
    y_test,
    y_prob
)

print("\nROC-AUC Score:", auc_score)


# =====================================
# 10. SAVE MODEL
# =====================================

joblib.dump(
    model,
    "models/xgboost_model.pkl"
)

print("\nXGBoost model saved successfully! ✅")