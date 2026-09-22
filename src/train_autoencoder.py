from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

MODELS_PATH = PROJECT_ROOT / "models"

MODELS_PATH.mkdir(exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

RANDOM_STATE = 42

NORMAL_TRAINING_SAMPLE = 500000

TEST_SAMPLE_SIZE = 200000


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")
print("Dataset path:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully! ✅")
print("Dataset shape:", df.shape)


# =========================================================
# SELECT REQUIRED FEATURES
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
    prefix="type",
    dtype=int
)


# =========================================================
# ENSURE EXPECTED COLUMNS
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


X = df[expected_columns]

y = df["isFraud"]


# =========================================================
# DISPLAY DISTRIBUTION
# =========================================================

print("\nFraud Distribution:")

print(y.value_counts())


# =========================================================
# TRAIN AUTOENCODER ON NORMAL TRANSACTIONS ONLY
# =========================================================

print("\nSelecting legitimate transactions...")

normal_data = X[y == 0]

print(
    "Total legitimate transactions:",
    len(normal_data)
)


# =========================================================
# SAMPLE NORMAL TRANSACTIONS
# =========================================================

sample_size = min(
    NORMAL_TRAINING_SAMPLE,
    len(normal_data)
)

print(
    f"\nUsing {sample_size:,} legitimate transactions "
    "for Autoencoder training..."
)


X_train = normal_data.sample(
    n=sample_size,
    random_state=RANDOM_STATE
)


# =========================================================
# SCALE FEATURES
# =========================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_train_scaled = X_train_scaled.astype(
    np.float32
)


# =========================================================
# BUILD AUTOENCODER
# =========================================================

print("\nBuilding Autoencoder...")


input_dim = X_train_scaled.shape[1]


input_layer = tf.keras.layers.Input(
    shape=(input_dim,)
)


# ---------------- ENCODER ----------------

encoded = tf.keras.layers.Dense(
    32,
    activation="relu"
)(input_layer)

encoded = tf.keras.layers.Dense(
    16,
    activation="relu"
)(encoded)

encoded = tf.keras.layers.Dense(
    8,
    activation="relu"
)(encoded)


# ---------------- DECODER ----------------

decoded = tf.keras.layers.Dense(
    16,
    activation="relu"
)(encoded)

decoded = tf.keras.layers.Dense(
    32,
    activation="relu"
)(decoded)

output_layer = tf.keras.layers.Dense(
    input_dim,
    activation="linear"
)(decoded)


# =========================================================
# CREATE MODEL
# =========================================================

autoencoder = tf.keras.Model(
    inputs=input_layer,
    outputs=output_layer
)


autoencoder.compile(
    optimizer="adam",
    loss="mse"
)


autoencoder.summary()


# =========================================================
# EARLY STOPPING
# =========================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)


# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTraining Autoencoder...")

history = autoencoder.fit(

    X_train_scaled,

    X_train_scaled,

    epochs=30,

    batch_size=1024,

    validation_split=0.1,

    shuffle=True,

    callbacks=[early_stopping],

    verbose=1

)


print("\nAutoencoder training completed successfully! ✅")


# =========================================================
# CALCULATE TRAINING RECONSTRUCTION ERROR
# =========================================================

print("\nCalculating normal reconstruction errors...")


train_predictions = autoencoder.predict(
    X_train_scaled,
    batch_size=1024,
    verbose=0
)


train_errors = np.mean(

    np.square(
        X_train_scaled - train_predictions
    ),

    axis=1

)


# =========================================================
# SET ANOMALY THRESHOLD
# =========================================================

# 99th percentile of normal transaction errors

threshold = np.percentile(
    train_errors,
    99
)


print(
    "\nAutoencoder Anomaly Threshold:",
    threshold
)


# =========================================================
# PREPARE TEST DATA
# =========================================================

print("\nPreparing evaluation dataset...")


test_size = min(
    TEST_SAMPLE_SIZE,
    len(X)
)


X_test = X.sample(

    n=test_size,

    random_state=123

)


y_test = y.loc[X_test.index]


print(
    "Test dataset size:",
    len(X_test)
)


print(
    "\nTest fraud distribution:"
)

print(
    y_test.value_counts()
)


# =========================================================
# SCALE TEST DATA
# =========================================================

X_test_scaled = scaler.transform(
    X_test
).astype(np.float32)


# =========================================================
# CALCULATE RECONSTRUCTION ERROR
# =========================================================

print("\nDetecting anomalies...")


test_predictions = autoencoder.predict(

    X_test_scaled,

    batch_size=1024,

    verbose=0

)


reconstruction_errors = np.mean(

    np.square(
        X_test_scaled - test_predictions
    ),

    axis=1

)


# =========================================================
# AUTOENCODER PREDICTIONS
# =========================================================

anomaly_predictions = (

    reconstruction_errors > threshold

).astype(int)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n================================================")
print("AUTOENCODER CLASSIFICATION REPORT")
print("================================================\n")


print(

    classification_report(

        y_test,

        anomaly_predictions,

        zero_division=0

    )

)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:")


cm = confusion_matrix(

    y_test,

    anomaly_predictions

)


print(cm)


# =========================================================
# ROC-AUC SCORE
# =========================================================

print("\nCalculating ROC-AUC...")


try:

    auc_score = roc_auc_score(

        y_test,

        reconstruction_errors

    )


    print(
        "\nROC-AUC Score:",
        auc_score
    )


except Exception as e:

    print(
        "\nCould not calculate ROC-AUC:",
        e
    )


# =========================================================
# SAVE AUTOENCODER
# =========================================================

print("\nSaving Autoencoder model...")


MODEL_PATH = (

    MODELS_PATH

    / "autoencoder_model.keras"

)


autoencoder.save(

    MODEL_PATH

)


# =========================================================
# SAVE SCALER
# =========================================================

SCALER_PATH = (

    MODELS_PATH

    / "autoencoder_scaler.pkl"

)


joblib.dump(

    scaler,

    SCALER_PATH

)


# =========================================================
# SAVE THRESHOLD
# =========================================================

THRESHOLD_PATH = (

    MODELS_PATH

    / "autoencoder_threshold.pkl"

)


joblib.dump(

    threshold,

    THRESHOLD_PATH

)


# =========================================================
# SAVE FEATURE COLUMNS
# =========================================================

FEATURES_PATH = (

    MODELS_PATH

    / "autoencoder_features.pkl"

)


joblib.dump(

    expected_columns,

    FEATURES_PATH

)


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n================================================")
print("AUTOENCODER MODEL SAVED SUCCESSFULLY! ✅")
print("================================================")


print("\nFiles created:")

print(MODEL_PATH)

print(SCALER_PATH)

print(THRESHOLD_PATH)

print(FEATURES_PATH)