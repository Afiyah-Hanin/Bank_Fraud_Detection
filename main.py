from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


# =========================================================
# CREATE FASTAPI APP
# =========================================================

app = FastAPI(
    title="Bank Fraud Detection API",
    description=(
        "Advanced Bank Fraud Detection System using "
        "Random Forest, Tuned XGBoost, Isolation Forest, "
        "Autoencoder and Graph Neural Network Fraud Ring Intelligence"
    ),
    version="4.0"
)


# =========================================================
# LOAD SUPERVISED MODELS
# =========================================================

print("Loading Random Forest model...")

random_forest_model = joblib.load(
    MODELS_DIR / "random_forest_model.pkl"
)


print("Loading Tuned XGBoost model...")

tuned_xgboost_model = joblib.load(
    MODELS_DIR / "tuned_xgboost_model.pkl"
)


# =========================================================
# LOAD ISOLATION FOREST
# =========================================================

print("Loading Isolation Forest model...")

isolation_forest_model = joblib.load(
    MODELS_DIR / "isolation_forest_model.pkl"
)

isolation_forest_scaler = joblib.load(
    MODELS_DIR / "isolation_forest_scaler.pkl"
)


# =========================================================
# LOAD AUTOENCODER
# =========================================================

print("Loading Autoencoder model...")

autoencoder_model = tf.keras.models.load_model(
    MODELS_DIR / "autoencoder_model.keras"
)

autoencoder_scaler = joblib.load(
    MODELS_DIR / "autoencoder_scaler.pkl"
)

autoencoder_threshold = float(
    joblib.load(
        MODELS_DIR / "autoencoder_threshold.pkl"
    )
)


# =========================================================
# LOAD GNN FRAUD RING RESULTS
# =========================================================

GNN_RESULTS_PATH = MODELS_DIR / "gnn_fraud_ring_results.csv"

print("Loading GNN Fraud Ring Intelligence results...")

if GNN_RESULTS_PATH.exists():

    gnn_fraud_rings = pd.read_csv(
        GNN_RESULTS_PATH
    )

    print(
        f"GNN fraud ring results loaded: "
        f"{len(gnn_fraud_rings)} groups"
    )

else:

    print(
        "WARNING: GNN fraud ring results file not found!"
    )

    gnn_fraud_rings = pd.DataFrame()


# =========================================================
# INPUT DATA FORMAT
# =========================================================

class Transaction(BaseModel):

    step: int

    transaction_type: str

    amount: float

    oldbalanceOrg: float

    newbalanceOrig: float

    oldbalanceDest: float

    newbalanceDest: float

    model_choice: str


# =========================================================
# CREATE INPUT FEATURES
# =========================================================

def create_input_data(transaction: Transaction):

    transaction_types = [

        "CASH_OUT",

        "DEBIT",

        "PAYMENT",

        "TRANSFER"

    ]

    encoded_types = {}

    for transaction_type in transaction_types:

        encoded_types[
            f"type_{transaction_type}"
        ] = (

            1

            if transaction.transaction_type
            == transaction_type

            else 0

        )


    input_data = pd.DataFrame([{

        "step":
            transaction.step,

        "amount":
            transaction.amount,

        "oldbalanceOrg":
            transaction.oldbalanceOrg,

        "newbalanceOrig":
            transaction.newbalanceOrig,

        "oldbalanceDest":
            transaction.oldbalanceDest,

        "newbalanceDest":
            transaction.newbalanceDest,

        "type_CASH_OUT":
            encoded_types["type_CASH_OUT"],

        "type_DEBIT":
            encoded_types["type_DEBIT"],

        "type_PAYMENT":
            encoded_types["type_PAYMENT"],

        "type_TRANSFER":
            encoded_types["type_TRANSFER"]

    }])


    return input_data


# =========================================================
# HOME API
# =========================================================

@app.get("/")
def home():

    return {

        "message":
            "Advanced Bank Fraud Detection API is running!",

        "version":
            "4.0",

        "available_transaction_models": [

            "Random Forest",

            "Tuned XGBoost",

            "Isolation Forest",

            "Autoencoder"

        ],

        "available_gnn_features": [

            "GraphSAGE Fraud Detection",

            "Fraud Ring Intelligence",

            "Critical Risk Ring Detection",

            "High Risk Ring Detection"

        ]

    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {

        "status": "healthy",

        "models": {

            "Random Forest": True,

            "Tuned XGBoost": True,

            "Isolation Forest": True,

            "Autoencoder": True,

            "GNN Fraud Ring Intelligence":
                not gnn_fraud_rings.empty

        }

    }


# =========================================================
# PREDICTION API
# =========================================================

@app.post("/predict")
def predict(transaction: Transaction):


    # =====================================================
    # VALIDATE TRANSACTION TYPE
    # =====================================================

    valid_transaction_types = [

        "CASH_OUT",

        "DEBIT",

        "PAYMENT",

        "TRANSFER"

    ]


    transaction.transaction_type = (
        transaction.transaction_type.upper()
    )


    if (
        transaction.transaction_type
        not in valid_transaction_types
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "Invalid transaction type. "
                "Use CASH_OUT, DEBIT, PAYMENT or TRANSFER."
            )

        )


    # =====================================================
    # CREATE INPUT DATA
    # =====================================================

    input_data = create_input_data(
        transaction
    )


    # =====================================================
    # RANDOM FOREST
    # =====================================================

    if transaction.model_choice == "Random Forest":

        fraud_probability = float(

            random_forest_model.predict_proba(
                input_data
            )[0][1]

        )


        threshold = 0.50


        prediction = (

            "FRAUD"

            if fraud_probability >= threshold

            else "LEGITIMATE"

        )


        return {

            "selected_model":
                "Random Forest",

            "prediction":
                prediction,

            "fraud_probability":
                round(
                    fraud_probability * 100,
                    2
                ),

            "threshold":
                threshold * 100

        }


    # =====================================================
    # TUNED XGBOOST
    # =====================================================

    elif transaction.model_choice == "Tuned XGBoost":

        fraud_probability = float(

            tuned_xgboost_model.predict_proba(
                input_data
            )[0][1]

        )


        threshold = 0.50


        prediction = (

            "FRAUD"

            if fraud_probability >= threshold

            else "LEGITIMATE"

        )


        return {

            "selected_model":
                "Tuned XGBoost",

            "prediction":
                prediction,

            "fraud_probability":
                round(
                    fraud_probability * 100,
                    2
                ),

            "threshold":
                threshold * 100

        }


    # =====================================================
    # ISOLATION FOREST
    # =====================================================

    elif transaction.model_choice == "Isolation Forest":


        scaled_data = (
            isolation_forest_scaler.transform(
                input_data
            )
        )


        anomaly_score = float(

            -isolation_forest_model.decision_function(
                scaled_data
            )[0]

        )


        raw_prediction = int(

            isolation_forest_model.predict(
                scaled_data
            )[0]

        )


        prediction = (

            "FRAUD"

            if raw_prediction == -1

            else "LEGITIMATE"

        )


        risk_score = max(

            0.0,

            min(

                100.0,

                anomaly_score * 100

            )

        )


        return {

            "selected_model":
                "Isolation Forest",

            "prediction":
                prediction,

            "anomaly_score":
                round(
                    anomaly_score,
                    6
                ),

            "fraud_probability":
                round(
                    risk_score,
                    2
                ),

            "threshold":
                "Anomaly Detection"

        }


    # =====================================================
    # AUTOENCODER
    # =====================================================

    elif transaction.model_choice == "Autoencoder":


        scaled_data = (
            autoencoder_scaler.transform(
                input_data
            )
        )


        reconstructed_data = (
            autoencoder_model.predict(
                scaled_data,
                verbose=0
            )
        )


        reconstruction_error = float(

            np.mean(

                np.square(

                    scaled_data
                    -
                    reconstructed_data

                )

            )

        )


        threshold_value = float(
            autoencoder_threshold
        )


        prediction = (

            "FRAUD"

            if reconstruction_error
            > threshold_value

            else "LEGITIMATE"

        )


        # Relative risk score

        risk_score = min(

            100.0,

            (
                reconstruction_error
                /
                threshold_value
            )
            * 50

        )


        return {

            "selected_model":
                "Autoencoder",

            "prediction":
                prediction,

            "fraud_probability":
                round(
                    float(risk_score),
                    2
                ),

            "reconstruction_error":
                round(
                    reconstruction_error,
                    6
                ),

            "threshold":
                round(
                    threshold_value,
                    6
                )

        }


    # =====================================================
    # INVALID MODEL
    # =====================================================

    else:

        raise HTTPException(

            status_code=400,

            detail={
                "error":
                    "Invalid model selected",

                "available_models": [

                    "Random Forest",

                    "Tuned XGBoost",

                    "Isolation Forest",

                    "Autoencoder"

                ]

            }

        )


# =========================================================
# GNN FRAUD RING SUMMARY
# =========================================================

@app.get("/gnn/summary")
def gnn_summary():

    if gnn_fraud_rings.empty:

        raise HTTPException(

            status_code=404,

            detail=(
                "GNN fraud ring results are not available. "
                "Run analyze_gnn_fraud_rings.py first."
            )

        )


    total_groups = int(
        len(gnn_fraud_rings)
    )


    critical_rings = int(

        (
            gnn_fraud_rings["risk_level"]
            == "CRITICAL"
        ).sum()

    )


    high_risk_rings = int(

        (
            gnn_fraud_rings["risk_level"]
            == "HIGH"
        ).sum()

    )


    medium_risk_rings = int(

        (
            gnn_fraud_rings["risk_level"]
            == "MEDIUM"
        ).sum()

    )


    low_risk_rings = int(

        (
            gnn_fraud_rings["risk_level"]
            == "LOW"
        ).sum()

    )


    predicted_fraud_accounts = int(

        gnn_fraud_rings[
            "predicted_fraud_accounts"
        ].sum()

    )


    return {

        "system":
            "GNN Fraud Ring Intelligence",

        "total_groups_analyzed":
            total_groups,

        "critical_risk_rings":
            critical_rings,

        "high_risk_rings":
            high_risk_rings,

        "medium_risk_rings":
            medium_risk_rings,

        "low_risk_rings":
            low_risk_rings,

        "total_predicted_fraud_accounts_in_groups":
            predicted_fraud_accounts

    }


# =========================================================
# GET FRAUD RINGS
# =========================================================

@app.get("/gnn/fraud-rings")
def get_fraud_rings(
    risk_level: str = None,
    limit: int = 20
):


    if gnn_fraud_rings.empty:

        raise HTTPException(

            status_code=404,

            detail="GNN fraud ring results not found."

        )


    results = gnn_fraud_rings.copy()


    # =====================================================
    # FILTER BY RISK LEVEL
    # =====================================================

    if risk_level is not None:

        risk_level = risk_level.upper()


        valid_risk_levels = [

            "CRITICAL",

            "HIGH",

            "MEDIUM",

            "LOW"

        ]


        if risk_level not in valid_risk_levels:

            raise HTTPException(

                status_code=400,

                detail=(
                    "Invalid risk level. "
                    "Use CRITICAL, HIGH, MEDIUM or LOW."
                )

            )


        results = results[

            results["risk_level"]
            == risk_level

        ]


    # =====================================================
    # SORT BY SUSPICION SCORE
    # =====================================================

    results = results.sort_values(

        by="suspicion_score",

        ascending=False

    )


    # =====================================================
    # LIMIT RESULTS
    # =====================================================

    limit = min(

        max(limit, 1),

        100

    )


    results = results.head(limit)


    return {

        "total_results":
            len(results),

        "risk_level_filter":
            risk_level,

        "fraud_rings":
            results.replace(
                {
                    np.nan: None
                }
            ).to_dict(
                orient="records"
            )

    }


# =========================================================
# GET SPECIFIC FRAUD RING
# =========================================================

@app.get("/gnn/fraud-rings/{ring_id}")
def get_fraud_ring(ring_id: int):


    if gnn_fraud_rings.empty:

        raise HTTPException(

            status_code=404,

            detail="GNN fraud ring results not found."

        )


    ring = gnn_fraud_rings[

        gnn_fraud_rings["ring_id"]
        == ring_id

    ]


    if ring.empty:

        raise HTTPException(

            status_code=404,

            detail=f"Fraud ring {ring_id} not found."

        )


    ring_data = (

        ring.iloc[0]
        .replace(
            {
                np.nan: None
            }
        )
        .to_dict()

    )


    return {

        "fraud_ring":
            ring_data

    }