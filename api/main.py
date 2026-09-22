from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Advanced Bank Fraud Detection API",
    description=(
        "Fraud detection using Tuned XGBoost, Random Forest, "
        "Isolation Forest and Autoencoder"
    ),
    version="1.0.0"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# MODEL PATHS
# ============================================================

XGB_PATH = MODELS_DIR / "tuned_xgboost_model.pkl"

RF_PATH = MODELS_DIR / "random_forest_model.pkl"

SCALER_PATH = MODELS_DIR / "scaler.pkl"

IF_PATH = MODELS_DIR / "isolation_forest_model.pkl"

IF_SCALER_PATH = MODELS_DIR / "isolation_forest_scaler.pkl"

AE_MODEL_PATH = MODELS_DIR / "autoencoder_model.keras"

AE_SCALER_PATH = MODELS_DIR / "autoencoder_scaler.pkl"

AE_FEATURES_PATH = MODELS_DIR / "autoencoder_features.pkl"

AE_THRESHOLD_PATH = MODELS_DIR / "autoencoder_threshold.pkl"


# ============================================================
# LOADING MODELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING ADVANCED FRAUD DETECTION MODELS")
print("=" * 70)


# ------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------

xgb_model = joblib.load(XGB_PATH)

print("✅ Tuned XGBoost loaded")


# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

rf_model = joblib.load(RF_PATH)

print("✅ Random Forest loaded")


# ------------------------------------------------------------
# SUPERVISED SCALER
# ------------------------------------------------------------

scaler = joblib.load(SCALER_PATH)

print("✅ Supervised model scaler loaded")


# ------------------------------------------------------------
# ISOLATION FOREST
# ------------------------------------------------------------

isolation_model = joblib.load(IF_PATH)

print("✅ Isolation Forest loaded")


# ------------------------------------------------------------
# ISOLATION FOREST SCALER
# ------------------------------------------------------------

isolation_scaler = joblib.load(IF_SCALER_PATH)

print("✅ Isolation Forest scaler loaded")


# ------------------------------------------------------------
# AUTOENCODER
# ------------------------------------------------------------

autoencoder = tf.keras.models.load_model(AE_MODEL_PATH)

print("✅ Autoencoder loaded")


# ------------------------------------------------------------
# AUTOENCODER SCALER
# ------------------------------------------------------------

autoencoder_scaler = joblib.load(AE_SCALER_PATH)

print("✅ Autoencoder scaler loaded")


# ------------------------------------------------------------
# AUTOENCODER FEATURES
# ------------------------------------------------------------

autoencoder_features = joblib.load(AE_FEATURES_PATH)

print("✅ Autoencoder features loaded")

print("Autoencoder features:")
print(autoencoder_features)


# ------------------------------------------------------------
# AUTOENCODER THRESHOLD
# ------------------------------------------------------------

autoencoder_threshold = joblib.load(AE_THRESHOLD_PATH)

# Sometimes threshold can be stored as numpy value
autoencoder_threshold = float(autoencoder_threshold)

print(f"✅ Autoencoder threshold: {autoencoder_threshold}")


print("=" * 70)
print("ALL MODELS LOADED SUCCESSFULLY! 🚀")
print("=" * 70)


# ============================================================
# EXACT MODEL FEATURE ORDER
# ============================================================

MODEL_FEATURES = [

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


# ============================================================
# API INPUT MODEL
# ============================================================

class Transaction(BaseModel):

    step: int

    amount: float

    oldbalanceOrg: float

    newbalanceOrig: float

    oldbalanceDest: float

    newbalanceDest: float

    transaction_type: str


# ============================================================
# TRANSACTION TYPE ENCODING
# ============================================================

def encode_transaction_type(transaction_type):

    transaction_type = transaction_type.upper().strip()

    allowed_types = [

        "CASH_IN",

        "CASH_OUT",

        "DEBIT",

        "PAYMENT",

        "TRANSFER"

    ]


    if transaction_type not in allowed_types:

        raise ValueError(

            f"Invalid transaction type: {transaction_type}. "
            f"Allowed values are: {allowed_types}"

        )


    # CASH_IN is the reference category.
    # Therefore it has all four encoded columns = 0.

    return {

        "type_CASH_OUT":

            1 if transaction_type == "CASH_OUT" else 0,


        "type_DEBIT":

            1 if transaction_type == "DEBIT" else 0,


        "type_PAYMENT":

            1 if transaction_type == "PAYMENT" else 0,


        "type_TRANSFER":

            1 if transaction_type == "TRANSFER" else 0

    }


# ============================================================
# CREATE EXACT FEATURE DATAFRAME
# ============================================================

def create_features(transaction):

    type_features = encode_transaction_type(
        transaction.transaction_type
    )


    data = {

        "step": transaction.step,

        "amount": transaction.amount,

        "oldbalanceOrg": transaction.oldbalanceOrg,

        "newbalanceOrig": transaction.newbalanceOrig,

        "oldbalanceDest": transaction.oldbalanceDest,

        "newbalanceDest": transaction.newbalanceDest,

        "type_CASH_OUT":

            type_features["type_CASH_OUT"],


        "type_DEBIT":

            type_features["type_DEBIT"],


        "type_PAYMENT":

            type_features["type_PAYMENT"],


        "type_TRANSFER":

            type_features["type_TRANSFER"]

    }


    df = pd.DataFrame([data])


    # Force exact feature order used during training

    df = df[MODEL_FEATURES]


    return df


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Advanced Bank Fraud Detection API is running! 🚀",


        "available_models": [

            "Tuned XGBoost",

            "Random Forest",

            "Isolation Forest",

            "Autoencoder"

        ],


        "endpoints": {

            "health": "/health",

            "models": "/models",

            "prediction": "/predict"

        }

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status": "healthy",

        "message":
            "Fraud Detection API is running successfully"

    }


# ============================================================
# MODEL STATUS
# ============================================================

@app.get("/models")
def model_status():

    return {

        "status": "All models loaded successfully",


        "models": {

            "tuned_xgboost": True,

            "random_forest": True,

            "isolation_forest": True,

            "autoencoder": True

        },


        "features":

            MODEL_FEATURES

    }


# ============================================================
# FRAUD PREDICTION
# ============================================================

@app.post("/predict")
def predict_fraud(transaction: Transaction):

    try:


        # ====================================================
        # CREATE FEATURES
        # ====================================================

        features = create_features(transaction)


        # ====================================================
        # XGBOOST PREDICTION
        # ====================================================

        # XGBoost was trained using these exact features.
        # We use the dataframe directly.

        xgb_probability = float(

            xgb_model.predict_proba(
                features
            )[0][1]

        )


        xgb_prediction = (

            "FRAUD"

            if xgb_probability >= 0.5

            else "LEGITIMATE"

        )


        # ====================================================
        # RANDOM FOREST PREDICTION
        # ====================================================

        rf_probability = float(

            rf_model.predict_proba(
                features
            )[0][1]

        )


        rf_prediction = (

            "FRAUD"

            if rf_probability >= 0.5

            else "LEGITIMATE"

        )


        # ====================================================
        # ISOLATION FOREST
        # ====================================================

        # Isolation Forest does not have feature_names_in_,
        # so we provide numeric values in the exact feature order.

        isolation_input = features.values


        isolation_scaled = isolation_scaler.transform(
            isolation_input
        )


        isolation_prediction = int(

            isolation_model.predict(
                isolation_scaled
            )[0]

        )


        isolation_score = float(

            isolation_model.decision_function(
                isolation_scaled
            )[0]

        )


        is_anomaly = (

            isolation_prediction == -1

        )


        # ====================================================
        # AUTOENCODER
        # ====================================================

        # Ensure exact feature order used during AE training

        ae_input = features[
            autoencoder_features
        ]


        ae_scaled = autoencoder_scaler.transform(
            ae_input
        )


        reconstructed = autoencoder.predict(
            ae_scaled,
            verbose=0
        )


        reconstruction_error = float(

            np.mean(

                np.square(

                    ae_scaled - reconstructed

                )

            )

        )


        autoencoder_anomaly = (

            reconstruction_error > autoencoder_threshold

        )


        # ====================================================
        # ENSEMBLE VOTING SYSTEM
        # ====================================================

        fraud_votes = 0


        if xgb_probability >= 0.5:

            fraud_votes += 1


        if rf_probability >= 0.5:

            fraud_votes += 1


        if is_anomaly:

            fraud_votes += 1


        if autoencoder_anomaly:

            fraud_votes += 1


        # ====================================================
        # FINAL DECISION
        # ====================================================

        if fraud_votes >= 3:

            final_prediction = "FRAUD"

            risk_level = "CRITICAL"


        elif fraud_votes == 2:

            final_prediction = "SUSPICIOUS"

            risk_level = "HIGH"


        elif fraud_votes == 1:

            final_prediction = "SUSPICIOUS"

            risk_level = "MEDIUM"


        else:

            final_prediction = "LEGITIMATE"

            risk_level = "LOW"


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "final_prediction":

                final_prediction,


            "risk_level":

                risk_level,


            "fraud_votes":

                fraud_votes,


            "transaction": {

                "step":

                    transaction.step,


                "amount":

                    transaction.amount,


                "transaction_type":

                    transaction.transaction_type.upper(),


                "oldbalanceOrg":

                    transaction.oldbalanceOrg,


                "newbalanceOrig":

                    transaction.newbalanceOrig,


                "oldbalanceDest":

                    transaction.oldbalanceDest,


                "newbalanceDest":

                    transaction.newbalanceDest

            },


            "model_results": {


                "tuned_xgboost": {

                    "fraud_probability":

                        round(
                            xgb_probability,
                            6
                        ),


                    "prediction":

                        xgb_prediction

                },


                "random_forest": {

                    "fraud_probability":

                        round(
                            rf_probability,
                            6
                        ),


                    "prediction":

                        rf_prediction

                },


                "isolation_forest": {

                    "anomaly":

                        bool(is_anomaly),


                    "anomaly_score":

                        round(
                            isolation_score,
                            6
                        )

                },


                "autoencoder": {

                    "anomaly":

                        bool(autoencoder_anomaly),


                    "reconstruction_error":

                        round(
                            reconstruction_error,
                            8
                        ),


                    "threshold":

                        round(
                            autoencoder_threshold,
                            8
                        )

                }

            }

        }


    except ValueError as error:

        raise HTTPException(

            status_code=400,

            detail=str(error)

        )


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)

        )