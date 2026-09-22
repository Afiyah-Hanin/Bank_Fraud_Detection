from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import joblib
import pandas as pd
import numpy as np


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
    version="4.1"
)


# =========================================================
# LAZY MODEL STORAGE
# =========================================================
# Models are NOT loaded when the server starts.
# They are loaded only when their endpoint is requested.

random_forest_model = None

tuned_xgboost_model = None

isolation_forest_model = None
isolation_forest_scaler = None

autoencoder_model = None
autoencoder_scaler = None
autoencoder_threshold = None

gnn_fraud_rings = None


# =========================================================
# MODEL LOADERS
# =========================================================

def get_random_forest_model():

    global random_forest_model

    if random_forest_model is None:

        print("Loading Random Forest model...")

        random_forest_model = joblib.load(
            MODELS_DIR / "random_forest_model.pkl"
        )

        print("Random Forest model loaded.")

    return random_forest_model


def get_tuned_xgboost_model():

    global tuned_xgboost_model

    if tuned_xgboost_model is None:

        print("Loading Tuned XGBoost model...")

        tuned_xgboost_model = joblib.load(
            MODELS_DIR / "tuned_xgboost_model.pkl"
        )

        print("Tuned XGBoost model loaded.")

    return tuned_xgboost_model


def get_isolation_forest():

    global isolation_forest_model
    global isolation_forest_scaler

    if isolation_forest_model is None:

        print("Loading Isolation Forest model...")

        isolation_forest_model = joblib.load(
            MODELS_DIR / "isolation_forest_model.pkl"
        )

        isolation_forest_scaler = joblib.load(
            MODELS_DIR / "isolation_forest_scaler.pkl"
        )

        print("Isolation Forest model loaded.")

    return (
        isolation_forest_model,
        isolation_forest_scaler
    )


def get_autoencoder():

    global autoencoder_model
    global autoencoder_scaler
    global autoencoder_threshold

    if autoencoder_model is None:

        print("Loading TensorFlow / Autoencoder...")

        # TensorFlow is imported ONLY when Autoencoder is used.
        import tensorflow as tf

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

        print("Autoencoder loaded.")

    return (
        autoencoder_model,
        autoencoder_scaler,
        autoencoder_threshold
    )


def get_gnn_fraud_rings():

    global gnn_fraud_rings

    if gnn_fraud_rings is None:

        results_path = (
            MODELS_DIR /
            "gnn_fraud_ring_results.csv"
        )

        print("Loading GNN fraud ring intelligence...")

        if results_path.exists():

            gnn_fraud_rings = pd.read_csv(
                results_path
            )

            print(
                "GNN fraud ring results loaded: "
                f"{len(gnn_fraud_rings)} groups"
            )

        else:

            print(
                "WARNING: GNN fraud ring results "
                "file not found."
            )

            gnn_fraud_rings = pd.DataFrame()

    return gnn_fraud_rings


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

    transaction_type = (
        transaction.transaction_type.upper()
    )

    transaction_types = [
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]

    if transaction_type not in transaction_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid transaction type. "
                "Use CASH_OUT, DEBIT, PAYMENT or TRANSFER."
            )
        )

    return pd.DataFrame([{

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
            int(transaction_type == "CASH_OUT"),

        "type_DEBIT":
            int(transaction_type == "DEBIT"),

        "type_PAYMENT":
            int(transaction_type == "PAYMENT"),

        "type_TRANSFER":
            int(transaction_type == "TRANSFER")

    }])


# =========================================================
# HOME API
# =========================================================

@app.get("/")
def home():

    return {

        "message":
            "Advanced Bank Fraud Detection API is running!",

        "version":
            "4.1",

        "deployment":
            "Render",

        "model_loading":
            "Lazy loading enabled",

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

    gnn_results_path = (
        MODELS_DIR /
        "gnn_fraud_ring_results.csv"
    )

    return {

        "status":
            "healthy",

        "model_loading":
            "lazy",

        "models_available": {

            "Random Forest":
                (
                    MODELS_DIR /
                    "random_forest_model.pkl"
                ).exists(),

            "Tuned XGBoost":
                (
                    MODELS_DIR /
                    "tuned_xgboost_model.pkl"
                ).exists(),

            "Isolation Forest":
                (
                    MODELS_DIR /
                    "isolation_forest_model.pkl"
                ).exists(),

            "Autoencoder":
                (
                    MODELS_DIR /
                    "autoencoder_model.keras"
                ).exists(),

            "GNN Fraud Ring Intelligence":
                gnn_results_path.exists()

        }

    }


# =========================================================
# PREDICTION API
# =========================================================

@app.post("/predict")
def predict(transaction: Transaction):

    transaction.transaction_type = (
        transaction.transaction_type.upper()
    )

    valid_transaction_types = [

        "CASH_OUT",

        "DEBIT",

        "PAYMENT",

        "TRANSFER"

    ]

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

    input_data = create_input_data(
        transaction
    )


    # =====================================================
    # RANDOM FOREST
    # =====================================================

    if transaction.model_choice == "Random Forest":

        model = get_random_forest_model()

        fraud_probability = float(

            model.predict_proba(
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
                50.0

        }


    # =====================================================
    # TUNED XGBOOST
    # =====================================================

    elif transaction.model_choice == "Tuned XGBoost":

        model = get_tuned_xgboost_model()

        fraud_probability = float(

            model.predict_proba(
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
                50.0

        }


    # =====================================================
    # ISOLATION FOREST
    # =====================================================

    elif transaction.model_choice == "Isolation Forest":

        (
            model,
            scaler
        ) = get_isolation_forest()

        scaled_data = scaler.transform(
            input_data
        )

        anomaly_score = float(

            -model.decision_function(
                scaled_data
            )[0]

        )

        raw_prediction = int(

            model.predict(
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

        (
            model,
            scaler,
            threshold_value
        ) = get_autoencoder()

        scaled_data = scaler.transform(
            input_data
        )

        reconstructed_data = model.predict(
            scaled_data,
            verbose=0
        )

        reconstruction_error = float(

            np.mean(

                np.square(

                    scaled_data -
                    reconstructed_data

                )

            )

        )

        prediction = (

            "FRAUD"

            if reconstruction_error >
            threshold_value

            else "LEGITIMATE"

        )

        if threshold_value > 0:

            risk_score = min(

                100.0,

                (
                    reconstruction_error /
                    threshold_value
                ) * 50

            )

        else:

            risk_score = 100.0


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
                    float(threshold_value),
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

    results = get_gnn_fraud_rings()

    if results.empty:

        raise HTTPException(

            status_code=404,

            detail=(
                "GNN fraud ring results are not available. "
                "Run analyze_gnn_fraud_rings.py first."
            )

        )

    total_groups = int(
        len(results)
    )

    critical_rings = int(

        (
            results["risk_level"]
            == "CRITICAL"
        ).sum()

    )

    high_risk_rings = int(

        (
            results["risk_level"]
            == "HIGH"
        ).sum()

    )

    medium_risk_rings = int(

        (
            results["risk_level"]
            == "MEDIUM"
        ).sum()

    )

    low_risk_rings = int(

        (
            results["risk_level"]
            == "LOW"
        ).sum()

    )

    predicted_fraud_accounts = int(

        results[
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

    results = get_gnn_fraud_rings()

    if results.empty:

        raise HTTPException(

            status_code=404,

            detail="GNN fraud ring results not found."

        )

    filtered_results = results.copy()

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

        filtered_results = filtered_results[

            filtered_results["risk_level"]
            == risk_level

        ]

    filtered_results = filtered_results.sort_values(

        by="suspicion_score",

        ascending=False

    )

    limit = min(

        max(limit, 1),

        100

    )

    filtered_results = filtered_results.head(
        limit
    )

    return {

        "total_results":
            len(filtered_results),

        "risk_level_filter":
            risk_level,

        "fraud_rings":
            filtered_results.replace(
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

    results = get_gnn_fraud_rings()

    if results.empty:

        raise HTTPException(

            status_code=404,

            detail="GNN fraud ring results not found."

        )

    ring = results[

        results["ring_id"]
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