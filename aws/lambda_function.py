import json
import os
import joblib
import numpy as np
import pandas as pd


# =========================================================
# MODEL PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# =========================================================
# LOAD MODELS
# =========================================================

RANDOM_FOREST_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)

XGBOOST_PATH = os.path.join(
    MODEL_DIR,
    "tuned_xgboost_model.pkl"
)


random_forest_model = joblib.load(
    RANDOM_FOREST_PATH
)

tuned_xgboost_model = joblib.load(
    XGBOOST_PATH
)


# =========================================================
# CREATE FEATURES
# =========================================================

def create_features(data):

    transaction_type = data[
        "transaction_type"
    ].upper()


    valid_types = [
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]


    if transaction_type not in valid_types:

        raise ValueError(
            "Invalid transaction_type. "
            "Use CASH_OUT, DEBIT, PAYMENT or TRANSFER."
        )


    features = pd.DataFrame([{

        "step":
            data["step"],

        "amount":
            data["amount"],

        "oldbalanceOrg":
            data["oldbalanceOrg"],

        "newbalanceOrig":
            data["newbalanceOrig"],

        "oldbalanceDest":
            data["oldbalanceDest"],

        "newbalanceDest":
            data["newbalanceDest"],

        "type_CASH_OUT":
            int(transaction_type == "CASH_OUT"),

        "type_DEBIT":
            int(transaction_type == "DEBIT"),

        "type_PAYMENT":
            int(transaction_type == "PAYMENT"),

        "type_TRANSFER":
            int(transaction_type == "TRANSFER")

    }])


    return features


# =========================================================
# PREDICTION
# =========================================================

def predict_transaction(data):

    model_choice = data.get(
        "model_choice",
        "Random Forest"
    )


    input_data = create_features(
        data
    )


    # =====================================================
    # RANDOM FOREST
    # =====================================================

    if model_choice == "Random Forest":

        probability = float(
            random_forest_model
            .predict_proba(input_data)[0][1]
        )


    # =====================================================
    # XGBOOST
    # =====================================================

    elif model_choice == "Tuned XGBoost":

        probability = float(
            tuned_xgboost_model
            .predict_proba(input_data)[0][1]
        )


    # =====================================================
    # INVALID MODEL
    # =====================================================

    else:

        raise ValueError(
            "AWS Lambda version currently supports "
            "Random Forest and Tuned XGBoost."
        )


    threshold = 0.50


    prediction = (

        "FRAUD"

        if probability >= threshold

        else "LEGITIMATE"

    )


    return {

        "selected_model":
            model_choice,

        "prediction":
            prediction,

        "fraud_probability":
            round(
                probability * 100,
                2
            ),

        "threshold":
            50.0

    }


# =========================================================
# AWS LAMBDA HANDLER
# =========================================================

def lambda_handler(event, context):

    try:

        # =================================================
        # API GATEWAY BODY
        # =================================================

        body = event.get(
            "body"
        )


        if body:

            if isinstance(body, str):

                data = json.loads(
                    body
                )

            else:

                data = body


        else:

            data = event


        # =================================================
        # PREDICT
        # =================================================

        result = predict_transaction(
            data
        )


        # =================================================
        # RESPONSE
        # =================================================

        return {

            "statusCode": 200,

            "headers": {

                "Content-Type":
                    "application/json",

                "Access-Control-Allow-Origin":
                    "*"

            },

            "body":
                json.dumps(result)

        }


    except Exception as e:

        return {

            "statusCode": 400,

            "headers": {

                "Content-Type":
                    "application/json",

                "Access-Control-Allow-Origin":
                    "*"

            },

            "body":
                json.dumps({

                    "error":
                        str(e)

                })

        }