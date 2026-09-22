from pathlib import Path

import joblib


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "tuned_xgboost_model.pkl"

tuned_xgboost_model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="Bank Fraud Detection - Vercel API",
    description="Serverless fraud detection API using Tuned XGBoost",
    version="1.0"
)


class Transaction(BaseModel):
    step: int
    transaction_type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


def create_input_data(transaction):

    transaction_type = transaction.transaction_type.upper()

    valid_types = [
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]

    if transaction_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid transaction type. "
                "Use CASH_OUT, DEBIT, PAYMENT or TRANSFER."
            )
        )

    values = [
        transaction.step,
        transaction.amount,
        transaction.oldbalanceOrg,
        transaction.newbalanceOrig,
        transaction.oldbalanceDest,
        transaction.newbalanceDest,
        int(transaction_type == "CASH_OUT"),
        int(transaction_type == "DEBIT"),
        int(transaction_type == "PAYMENT"),
        int(transaction_type == "TRANSFER")
    ]

    return [values]


@app.get("/")
def home():

    return {
        "message": "Bank Fraud Detection Vercel API is running",
        "deployment": "Vercel Serverless",
        "model": "Tuned XGBoost",
        "status": "online"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "Tuned XGBoost",
        "model_loaded": True
    }


@app.post("/predict")
def predict(transaction: Transaction):

    input_data = create_input_data(transaction)

    probability = float(
        tuned_xgboost_model.predict_proba(input_data)[0][1]
    )

    threshold = 0.50

    prediction = (
        "FRAUD"
        if probability >= threshold
        else "LEGITIMATE"
    )

    return {
        "selected_model": "Tuned XGBoost",
        "prediction": prediction,
        "fraud_probability": round(probability * 100, 2),
        "threshold": 50.0
    }