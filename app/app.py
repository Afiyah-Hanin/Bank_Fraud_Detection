import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

API_URL = "https://bank-fraud-detection-1zoy.onrender.com"
PREDICT_URL = f"{API_URL}/predict"

st.set_page_config(
    page_title="Bank Fraud Detection System",
    page_icon="🏦",
    layout="wide",
)

# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        color: #b0b0b0;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================

@st.cache_resource
def load_models():
    rf = joblib.load("models/random_forest_model.pkl")
    xgb = joblib.load("models/tuned_xgboost_model.pkl")
    return rf, xgb


def make_input_dataframe(
    step,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    transaction_type,
):
    return pd.DataFrame(
        [[
            step,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,
            1 if transaction_type == "CASH_OUT" else 0,
            1 if transaction_type == "DEBIT" else 0,
            1 if transaction_type == "PAYMENT" else 0,
            1 if transaction_type == "TRANSFER" else 0,
        ]],
        columns=[
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "type_CASH_OUT",
            "type_DEBIT",
            "type_PAYMENT",
            "type_TRANSFER",
        ],
    )


def show_shap(selected_model, input_data):
    st.divider()
    st.subheader("🧠 SHAP Explainability")
    st.write("Features influencing the tree-model prediction:")

    try:
        with st.spinner("Generating SHAP explanation..."):
            explainer = shap.TreeExplainer(selected_model)
            shap_values = explainer(input_data)

            values = np.asarray(shap_values.values)

            if values.ndim == 3:
                fraud_values = values[0, :, 1]
                base_values = np.asarray(shap_values.base_values)
                fraud_base = float(base_values[0, 1])
            elif values.ndim == 2:
                fraud_values = values[0]
                base_values = np.asarray(shap_values.base_values)
                fraud_base = float(base_values.flatten()[0])
            else:
                raise ValueError(f"Unexpected SHAP shape: {values.shape}")

            explanation = shap.Explanation(
                values=fraud_values,
                base_values=fraud_base,
                data=input_data.iloc[0].values,
                feature_names=input_data.columns.tolist(),
            )

            st.write("### 🔍 Why did the model make this prediction?")
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation, max_display=10, show=False)
            st.pyplot(fig, clear_figure=True)

            shap_df = pd.DataFrame({
                "Feature": input_data.columns,
                "SHAP Impact": fraud_values,
            })
            shap_df["Absolute Impact"] = shap_df["SHAP Impact"].abs()
            shap_df["Impact Direction"] = np.where(
                shap_df["SHAP Impact"] > 0,
                "🚨 Toward Fraud",
                "✅ Toward Legitimate",
            )
            shap_df = shap_df.sort_values(
                "Absolute Impact", ascending=False
            )

            st.write("### 📋 Feature Contributions")
            st.dataframe(
                shap_df[
                    ["Feature", "SHAP Impact", "Impact Direction"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.write("### 📊 Feature Impact")
            chart_df = shap_df.sort_values("SHAP Impact")
            st.bar_chart(
                chart_df.set_index("Feature")["SHAP Impact"]
            )

            top = shap_df.iloc[0]
            if top["SHAP Impact"] > 0:
                st.warning(
                    f"Strongest feature: **{top['Feature']}** — "
                    "pushing the prediction toward fraud."
                )
            else:
                st.success(
                    f"Strongest feature: **{top['Feature']}** — "
                    "pushing the prediction toward legitimate."
                )

    except Exception as e:
        st.error("❌ SHAP explanation could not be generated.")
        st.code(str(e))


def show_gnn_intelligence():
    st.divider()
    st.subheader("🕸️ GNN Fraud-Ring Intelligence")

    try:
        response = requests.get(
            f"{API_URL}/gnn/summary",
            timeout=30
        )

        if response.status_code != 200:
            st.warning(
                f"GNN summary unavailable: HTTP {response.status_code}"
            )
            st.code(response.text)
            return

        summary = response.json()

        total_groups = summary.get(
            "total_groups_analyzed", 0
        )

        predicted_fraud = summary.get(
            "total_predicted_fraud_accounts_in_groups", 0
        )

        critical = summary.get(
            "critical_risk_rings", 0
        )

        high = summary.get(
            "high_risk_rings", 0
        )

        medium = summary.get(
            "medium_risk_rings", 0
        )

        low = summary.get(
            "low_risk_rings", 0
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Fraud Accounts",
                f"{predicted_fraud:,}"
            )

        with c2:
            st.metric(
                "Critical Rings",
                f"{critical:,}"
            )

        with c3:
            st.metric(
                "High-Risk Rings",
                f"{high:,}"
            )

        with c4:
            st.metric(
                "Groups Analyzed",
                f"{total_groups:,}"
            )

        st.write("### 📊 Fraud-Ring Risk Distribution")

        ring_df = pd.DataFrame({
            "Risk Level": [
                "Critical",
                "High",
                "Medium",
                "Low"
            ],
            "Groups": [
                critical,
                high,
                medium,
                low
            ]
        })

        st.bar_chart(
            ring_df.set_index("Risk Level")
        )

        st.info(
            "GNN ring categories are project-defined intelligence "
            "signals and should be treated as analyst leads, "
            "not confirmed fraud labels."
        )

    except requests.exceptions.RequestException as e:
        st.warning(f"GNN service unavailable: {e}")

    except Exception as e:
        st.error("❌ Error displaying GNN intelligence.")
        st.code(str(e))


    
  


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🏦 Bank Fraud Detection System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "AI-Powered Real-Time Bank Fraud Detection"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()

# =========================================================
# SIDEBAR / API STATUS
# =========================================================

with st.sidebar:
    st.header("⚙️ System Status")

    try:
        health = requests.get(
            f"{API_URL}/health",
            timeout=15,
        )

        if health.status_code == 200:
            st.success("🟢 Render API Online")
        else:
            st.warning(
                f"🟡 API responded with HTTP {health.status_code}"
            )
    except requests.exceptions.RequestException:
        st.error("🔴 API unreachable")

    st.caption("Live API")
    st.code(API_URL)

# =========================================================
# LOAD LOCAL TREE MODELS FOR SHAP
# =========================================================

try:
    random_forest_model, tuned_xgboost_model = load_models()
except Exception as e:
    st.error("❌ Error loading local models used for SHAP.")
    st.code(str(e))
    st.stop()

# =========================================================
# MODEL SELECTION
# =========================================================

st.subheader("🤖 Select Detection Model")

model_choice = st.selectbox(
    "Choose the model you want to use:",
    [
        "Random Forest",
        "Tuned XGBoost",
        "Isolation Forest",
        "Autoencoder",
    ],
)

selected_model = None

if model_choice == "Random Forest":
    selected_model = random_forest_model
elif model_choice == "Tuned XGBoost":
    selected_model = tuned_xgboost_model

st.success(f"Selected Model: {model_choice}")

st.divider()

# =========================================================
# TRANSACTION INPUT
# =========================================================

st.subheader("💳 Transaction Details")

c1, c2 = st.columns(2)

with c1:
    step = st.number_input(
        "Transaction Step (Hour)",
        min_value=0,
        value=1,
        step=1,
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"],
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0,
        step=100.0,
    )

    oldbalanceOrg = st.number_input(
        "Sender Old Balance",
        min_value=0.0,
        value=1000.0,
        step=100.0,
    )

with c2:
    newbalanceOrig = st.number_input(
        "Sender New Balance",
        min_value=0.0,
        value=0.0,
        step=100.0,
    )

    oldbalanceDest = st.number_input(
        "Receiver Old Balance",
        min_value=0.0,
        value=0.0,
        step=100.0,
    )

    newbalanceDest = st.number_input(
        "Receiver New Balance",
        min_value=0.0,
        value=0.0,
        step=100.0,
    )

# =========================================================
# CONSISTENCY CHECK
# =========================================================

st.divider()
st.subheader("🔍 Transaction Consistency Check")

sender_difference = oldbalanceOrg - newbalanceOrig
receiver_difference = newbalanceDest - oldbalanceDest

sender_matches = abs(sender_difference - amount) < 0.01
receiver_matches = True

c1, c2 = st.columns(2)

with c1:
    st.write("### Sender Check")
    st.write(
        f"Money deducted from sender: "
        f"₹ {sender_difference:,.2f}"
    )

    if sender_matches:
        st.success("✅ Sender balance matches transaction amount")
    else:
        st.warning(
            "⚠️ Sender balance does not exactly match "
            "the transaction amount."
        )

with c2:
    st.write("### Receiver Check")

    if transaction_type == "PAYMENT":
        st.info(
            "ℹ️ PAYMENT transactions in PaySim may not "
            "show receiver balance updates."
        )
    elif transaction_type == "TRANSFER":
        st.write(
            f"Money received by receiver: "
            f"₹ {receiver_difference:,.2f}"
        )

        receiver_matches = (
            abs(receiver_difference - amount) < 0.01
        )

        if receiver_matches:
            st.success(
                "✅ Receiver balance matches transaction amount"
            )
        else:
            st.warning(
                "⚠️ Receiver balance does not exactly match "
                "the transaction amount."
            )
    elif transaction_type == "CASH_OUT":
        st.info(
            "ℹ️ CASH_OUT represents a withdrawal; "
            "destination balance behavior may differ."
        )
    else:
        st.info(
            "ℹ️ DEBIT transactions can have different "
            "destination balance behavior."
        )

# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔍 Predict Transaction",
    type="primary",
    use_container_width=True,
):
    input_data = make_input_dataframe(
        step,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        transaction_type,
    )

    transaction_data = {
        "step": step,
        "transaction_type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "model_choice": model_choice,
    }

    try:
        with st.spinner("Analyzing transaction..."):
            response = requests.post(
                PREDICT_URL,
                json=transaction_data,
                timeout=60,
            )

        if response.status_code != 200:
            st.error(
                f"❌ API Error: {response.status_code}"
            )
            st.code(response.text)
            st.stop()

        result = response.json()

        if "error" in result:
            st.error(f"❌ {result['error']}")
            st.stop()

        st.divider()
        st.subheader("📊 Prediction Result")

        prediction = str(
            result.get("prediction", "UNKNOWN")
        ).upper()

        # =====================================================
        # TREE MODELS
        # =====================================================

        if model_choice in ["Random Forest", "Tuned XGBoost"]:
            fraud_probability = float(
                result.get("fraud_probability", 0.0)
            )
            threshold = float(
                result.get("threshold", 50)
            )

            if fraud_probability >= 80:
                risk_level = "CRITICAL"
            elif fraud_probability >= 60:
                risk_level = "HIGH"
            elif fraud_probability >= 30:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Fraud Probability",
                    f"{fraud_probability:.2f}%",
                )
            with c2:
                st.metric(
                    "Threshold",
                    f"{threshold:.0f}%",
                )
            with c3:
                st.metric(
                    "Risk Level",
                    risk_level,
                )
            with c4:
                st.metric(
                    "Model",
                    model_choice,
                )

            if prediction == "FRAUD":
                st.error(
                    "🚨 FRAUD ALERT — SUSPICIOUS TRANSACTION DETECTED"
                )
                st.write(
                    "The selected model classified this transaction "
                    "as fraud based on the configured decision threshold."
                )
            else:
                st.success(
                    "✅ TRANSACTION CLASSIFIED AS LEGITIMATE"
                )
                st.write(
                    "The selected model classified this transaction "
                    "below the configured fraud threshold."
                )

            if (
                transaction_type == "TRANSFER"
                and not receiver_matches
            ):
                st.warning(
                    "⚠️ Analyst Alert: receiver balance is "
                    "inconsistent with the transaction amount."
                )
                st.info(
                    "This consistency issue is an additional signal "
                    "and does not by itself confirm fraud."
                )

            # Analyst summary
            st.write("### 🧾 Analyst Alert Summary")
            alert_df = pd.DataFrame([{
                "Transaction Type": transaction_type,
                "Amount": f"₹ {amount:,.2f}",
                "Model": model_choice,
                "Prediction": prediction,
                "Fraud Probability": f"{fraud_probability:.2f}%",
                "Risk Level": risk_level,
                "Sender Check": "PASS" if sender_matches else "WARNING",
                "Receiver Check": (
                    "PASS"
                    if receiver_matches
                    else "WARNING"
                ),
            }])
            st.dataframe(
                alert_df,
                use_container_width=True,
                hide_index=True,
            )

            show_shap(selected_model, input_data)

        # =====================================================
        # ISOLATION FOREST
        # =====================================================

        elif model_choice == "Isolation Forest":
            anomaly_score = result.get("anomaly_score")
            fraud_probability = result.get("fraud_probability")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Detection Result", prediction)
            with c2:
                st.metric(
                    "Anomaly Score",
                    (
                        f"{float(anomaly_score):.6f}"
                        if anomaly_score is not None
                        else "N/A"
                    ),
                )
            with c3:
                st.metric("Model", "Isolation Forest")

            if prediction == "FRAUD":
                st.error(
                    "🚨 ANOMALOUS TRANSACTION DETECTED"
                )
            else:
                st.success(
                    "✅ NORMAL TRANSACTION PATTERN"
                )

            if fraud_probability is not None:
                st.metric(
                    "Relative Risk Score",
                    f"{float(fraud_probability):.2f}%",
                )

            st.info(
                "Isolation Forest is an unsupervised anomaly "
                "detection model. An anomaly is not automatically "
                "confirmed fraud."
            )

        # =====================================================
        # AUTOENCODER
        # =====================================================

        elif model_choice == "Autoencoder":
            fraud_probability = float(
                result.get("fraud_probability", 0.0)
            )
            reconstruction_error = result.get(
                "reconstruction_error"
            )
            threshold = result.get("threshold")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Risk Score",
                    f"{fraud_probability:.2f}%",
                )
            with c2:
                st.metric(
                    "Reconstruction Error",
                    (
                        f"{float(reconstruction_error):.6f}"
                        if reconstruction_error is not None
                        else "N/A"
                    ),
                )
            with c3:
                st.metric(
                    "Model",
                    "Autoencoder",
                )

            if prediction == "FRAUD":
                st.error(
                    "🚨 SUSPICIOUS TRANSACTION DETECTED"
                )
            else:
                st.success(
                    "✅ NORMAL TRANSACTION PATTERN"
                )

            if threshold is not None:
                st.write(
                    f"**Detection Threshold:** "
                    f"`{float(threshold):.6f}`"
                )

            st.info(
                "The Autoencoder detects unusual transactions "
                "using reconstruction error."
            )

        # =====================================================
        # GNN INTELLIGENCE
        # =====================================================

        show_gnn_intelligence()

    except requests.exceptions.Timeout:
        st.error(
            "❌ The Render API request timed out. "
            "The service may be waking up or loading a model."
        )
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Could not connect to the Render API."
        )
    except requests.exceptions.RequestException as e:
        st.error("❌ API request failed.")
        st.code(str(e))
    except Exception as e:
        st.error("❌ Unexpected dashboard error.")
        st.code(str(e))

# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "🏦 Bank Fraud Detection System | "
    "Random Forest + Tuned XGBoost + Isolation Forest + "
    "Autoencoder + GNN Fraud-Ring Intelligence | "
    "FastAPI + Render + SHAP"
)
