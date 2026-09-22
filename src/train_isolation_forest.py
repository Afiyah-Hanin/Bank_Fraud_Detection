from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# =========================================================
# LOAD DATASET
# =========================================================

print("Loading dataset...")
print("Dataset path:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully! ✅")
print("Dataset shape:", df.shape)


# =========================================================
# SELECT REQUIRED COLUMNS
# =========================================================

features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

df = df[features + ["isFraud"]].copy()


# =========================================================
# ENCODE TRANSACTION TYPE
# =========================================================

print("\nEncoding transaction types...")

df = pd.get_dummies(
    df,
    columns=["type"],
    prefix="type"
)


# =========================================================
# ENSURE ALL EXPECTED COLUMNS EXIST
# =========================================================

expected_columns = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]

for column in expected_columns:

    if column not in df.columns:
        df[column] = 0


# =========================================================
# PREPARE FEATURES AND TARGET
# =========================================================

X = df[expected_columns]

y = df["isFraud"]


# =========================================================
# DISPLAY FRAUD DISTRIBUTION
# =========================================================

print("\nActual Fraud Distribution:")

print(y.value_counts())


# =========================================================
# SAMPLE DATA FOR TRAINING
# =========================================================

sample_size = min(500000, len(X))

print(
    f"\nUsing {sample_size:,} transactions "
    "for Isolation Forest training..."
)

X_sample = X.sample(
    n=sample_size,
    random_state=42
)


# =========================================================
# SCALE FEATURES
# =========================================================

print("\nScaling features...")

scaler = StandardScaler()

X_sample_scaled = scaler.fit_transform(
    X_sample
)


# =========================================================
# TRAIN ISOLATION FOREST
# =========================================================

print("\nTraining Isolation Forest...")

isolation_forest = IsolationForest(

    n_estimators=200,

    contamination=0.01,

    random_state=42,

    n_jobs=-1

)


isolation_forest.fit(
    X_sample_scaled
)

print(
    "Isolation Forest training completed successfully! ✅"
)


# =========================================================
# EVALUATION DATA
# =========================================================

print("\nPreparing evaluation data...")

test_size = min(200000, len(X))

X_test = X.sample(
    n=test_size,
    random_state=123
)

y_test = y.loc[X_test.index]


X_test_scaled = scaler.transform(
    X_test
)


# =========================================================
# MAKE ANOMALY PREDICTIONS
# =========================================================

print("\nDetecting anomalies...")

predictions = isolation_forest.predict(
    X_test_scaled
)


# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

anomaly_predictions = np.where(
    predictions == -1,
    1,
    0
)


# =========================================================
# RESULTS
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        anomaly_predictions,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    anomaly_predictions
)

print(cm)


# =========================================================
# ROC-AUC USING ANOMALY SCORES
# =========================================================

print("\nCalculating anomaly scores...")

# Higher score = more anomalous

anomaly_scores = -isolation_forest.decision_function(
    X_test_scaled
)


try:

    auc_score = roc_auc_score(
        y_test,
        anomaly_scores
    )

    print("\nROC-AUC Score:", auc_score)

except Exception as e:

    print(
        "\nCould not calculate ROC-AUC:"
    )

    print(e)


# =========================================================
# SAVE MODEL
# =========================================================

print("\nSaving Isolation Forest model...")

MODEL_PATH = (
    MODEL_DIR
    / "isolation_forest_model.pkl"
)

SCALER_PATH = (
    MODEL_DIR
    / "isolation_forest_scaler.pkl"
)


joblib.dump(
    isolation_forest,
    MODEL_PATH
)


joblib.dump(
    scaler,
    SCALER_PATH
)


# =========================================================
# SUCCESS
# =========================================================

print("\n" + "=" * 55)

print(
    "Isolation Forest model saved successfully! ✅"
)

print("=" * 55)

print("\nModel saved at:")

print(MODEL_PATH)

print("\nScaler saved at:")

print(SCALER_PATH)