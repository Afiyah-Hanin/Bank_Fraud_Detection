import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# =========================================================
# API URL
# =========================================================

API_URL = "http://127.0.0.1:8000/predict"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Bank Fraud Detection System",
    page_icon="🏦",
    layout="wide"
)


# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #b0b0b0;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🏦 Bank Fraud Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Real-Time Bank Fraud Detection'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# LOAD MODELS FOR SHAP
# =========================================================

@st.cache_resource
def load_models():

    random_forest_model = joblib.load(
        "models/random_forest_model.pkl"
    )

    tuned_xgboost_model = joblib.load(
        "models/tuned_xgboost_model.pkl"
    )

    return (
        random_forest_model,
        tuned_xgboost_model
    )


try:

    (
        random_forest_model,
        tuned_xgboost_model

    ) = load_models()


except Exception as e:

    st.error("❌ Error loading models.")

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
        "Autoencoder"
    ]
)


# =========================================================
# SELECT MODEL FOR SHAP
# =========================================================

selected_model = None


if model_choice == "Random Forest":

    selected_model = random_forest_model


elif model_choice == "Tuned XGBoost":

    selected_model = tuned_xgboost_model


st.success(
    f"Selected Model: {model_choice}"
)

st.divider()


# =========================================================
# TRANSACTION INPUT
# =========================================================

st.subheader("💳 Transaction Details")


step = st.number_input(
    "Transaction Step (Hour)",
    min_value=0,
    value=1
)


transaction_type = st.selectbox(
    "Transaction Type",
    [
        "TRANSFER",
        "CASH_OUT",
        "PAYMENT",
        "DEBIT"
    ]
)


amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)


oldbalanceOrg = st.number_input(
    "Sender Old Balance",
    min_value=0.0,
    value=1000.0
)


newbalanceOrig = st.number_input(
    "Sender New Balance",
    min_value=0.0,
    value=0.0
)


oldbalanceDest = st.number_input(
    "Receiver Old Balance",
    min_value=0.0,
    value=0.0
)


newbalanceDest = st.number_input(
    "Receiver New Balance",
    min_value=0.0,
    value=0.0
)


st.divider()


# =========================================================
# TRANSACTION CONSISTENCY CHECK
# =========================================================

st.subheader("🔍 Transaction Consistency Check")


sender_difference = (
    oldbalanceOrg - newbalanceOrig
)

receiver_difference = (
    newbalanceDest - oldbalanceDest
)

receiver_matches = True


# =========================================================
# SENDER CHECK
# =========================================================

st.write("### Sender Check")


st.write(
    f"Money deducted from sender: "
    f"₹ {sender_difference:,.2f}"
)


if abs(sender_difference - amount) < 0.01:

    st.success(
        "✅ Sender balance matches transaction amount"
    )

else:

    st.warning(
        "⚠️ Sender balance does not exactly match "
        "the transaction amount"
    )


# =========================================================
# RECEIVER CHECK
# =========================================================

st.write("### Receiver Check")


if transaction_type == "PAYMENT":

    st.info(
        "ℹ️ PAYMENT transactions in the PaySim dataset "
        "may not show receiver balance updates."
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
        "ℹ️ CASH_OUT represents a withdrawal. "
        "Destination balance behavior may differ."
    )


elif transaction_type == "DEBIT":

    st.info(
        "ℹ️ DEBIT transactions can have different "
        "destination balance behavior."
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("🔍 Predict Transaction"):


    # =====================================================
    # CREATE INPUT DATA FOR SHAP
    # =====================================================

    input_data = pd.DataFrame(

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

            1 if transaction_type == "TRANSFER" else 0

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

            "type_TRANSFER"

        ]

    )


    # =====================================================
    # DATA SENT TO FASTAPI
    # =====================================================

    transaction_data = {

        "step": step,

        "transaction_type": transaction_type,

        "amount": amount,

        "oldbalanceOrg": oldbalanceOrg,

        "newbalanceOrig": newbalanceOrig,

        "oldbalanceDest": oldbalanceDest,

        "newbalanceDest": newbalanceDest,

        "model_choice": model_choice

    }


    # =====================================================
    # CALL FASTAPI
    # =====================================================

    try:

        response = requests.post(

            API_URL,

            json=transaction_data,

            timeout=30

        )


        # =================================================
        # SUCCESSFUL RESPONSE
        # =================================================

        if response.status_code == 200:


            result = response.json()


            # Check API error

            if "error" in result:

                st.error(
                    f"❌ {result['error']}"
                )

                st.stop()


            st.divider()

            st.subheader("📊 Prediction Result")


            # =================================================
            # RANDOM FOREST / XGBOOST
            # =================================================

            if model_choice in [

                "Random Forest",

                "Tuned XGBoost"

            ]:


                fraud_probability = float(

                    result.get(
                        "fraud_probability",
                        0.0
                    )

                )


                prediction = result.get(

                    "prediction",

                    "UNKNOWN"

                )


                threshold = result.get(

                    "threshold",

                    50

                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(

                        "Fraud Probability",

                        f"{fraud_probability:.2f}%"

                    )


                with col2:

                    st.metric(

                        "Decision Threshold",

                        f"{float(threshold):.0f}%"

                    )


                with col3:

                    st.metric(

                        "Selected Model",

                        model_choice

                    )


                # =============================================
                # FRAUD RESULT
                # =============================================

                if prediction.upper() == "FRAUD":


                    st.error(
                        "🚨 FRAUDULENT TRANSACTION DETECTED"
                    )


                    st.write(
                        "The model predicts that this transaction "
                        "has a high probability of being fraudulent."
                    )


                else:


                    st.success(
                        "✅ LEGITIMATE TRANSACTION"
                    )


                    st.write(
                        "The model predicts that this transaction "
                        "is likely legitimate."
                    )


                # =============================================
                # CONSISTENCY WARNING
                # =============================================

                if (

                    transaction_type == "TRANSFER"

                    and not receiver_matches

                ):


                    st.warning(
                        "⚠️ Important: The receiver balance is "
                        "inconsistent with the transaction amount."
                    )


                    st.info(
                        "This inconsistency does not automatically "
                        "mean the transaction is fraudulent."
                    )


                # =================================================
                # SHAP EXPLANATION
                # =================================================

                st.divider()

                st.subheader("🧠 SHAP Explanation")


                st.write(
                    "This section explains which features "
                    "influenced the fraud prediction."
                )


                try:


                    with st.spinner(

                        "Generating SHAP explanation..."

                    ):


                        explainer = shap.TreeExplainer(
                            selected_model
                        )


                        shap_values = explainer(
                            input_data
                        )


                        values = shap_values.values


                        # =========================================
                        # HANDLE SHAP OUTPUT
                        # =========================================

                        if len(values.shape) == 3:


                            fraud_values = values[0, :, 1]


                            base_array = np.array(
                                shap_values.base_values
                            )


                            fraud_base_value = (
                                base_array[0, 1]
                            )


                        elif len(values.shape) == 2:


                            fraud_values = values[0]


                            base_array = np.array(
                                shap_values.base_values
                            )


                            fraud_base_value = (
                                base_array.flatten()[0]
                            )


                        else:


                            raise ValueError(

                                f"Unexpected SHAP shape: "
                                f"{values.shape}"

                            )


                        # =========================================
                        # CREATE SHAP EXPLANATION
                        # =========================================

                        fraud_explanation = shap.Explanation(


                            values=fraud_values,


                            base_values=fraud_base_value,


                            data=input_data.iloc[0].values,


                            feature_names=(
                                input_data.columns.tolist()
                            )

                        )


                        # =========================================
                        # WATERFALL PLOT
                        # =========================================

                        st.write(
                            "### 🔍 Why did the model make this prediction?"
                        )


                        plt.figure(
                            figsize=(10, 6)
                        )


                        shap.plots.waterfall(

                            fraud_explanation,

                            max_display=10,

                            show=False

                        )


                        st.pyplot(

                            plt.gcf(),

                            clear_figure=True

                        )


                        # =========================================
                        # FEATURE CONTRIBUTIONS
                        # =========================================

                        st.write(
                            "### 📋 Feature Contributions"
                        )


                        shap_df = pd.DataFrame({

                            "Feature": input_data.columns,

                            "SHAP Impact": fraud_values

                        })


                        shap_df[
                            "Absolute Impact"
                        ] = (

                            shap_df[
                                "SHAP Impact"
                            ].abs()

                        )


                        shap_df[
                            "Impact Direction"
                        ] = np.where(


                            shap_df[
                                "SHAP Impact"
                            ] > 0,


                            "🚨 Pushes Toward Fraud",


                            "✅ Pushes Toward Legitimate"

                        )


                        shap_df = shap_df.sort_values(

                            by="Absolute Impact",

                            ascending=False

                        )


                        st.dataframe(

                            shap_df[

                                [

                                    "Feature",

                                    "SHAP Impact",

                                    "Impact Direction"

                                ]

                            ],

                            use_container_width=True,

                            hide_index=True

                        )


                        # =========================================
                        # FEATURE IMPACT CHART
                        # =========================================

                        st.write(
                            "### 📊 Feature Impact Visualization"
                        )


                        chart_df = shap_df.sort_values(

                            by="SHAP Impact"

                        )


                        st.bar_chart(

                            chart_df.set_index(
                                "Feature"
                            )["SHAP Impact"]

                        )


                        # =========================================
                        # MAIN EXPLANATION
                        # =========================================

                        top_feature = shap_df.iloc[0]


                        st.write(
                            "### 💡 Main Explanation"
                        )


                        if top_feature[
                            "SHAP Impact"
                        ] > 0:


                            st.warning(

                                f"The strongest feature is "
                                f"**{top_feature['Feature']}**. "

                                f"It is pushing the prediction "
                                f"toward **FRAUD**."

                            )


                        else:


                            st.success(

                                f"The strongest feature is "
                                f"**{top_feature['Feature']}**. "

                                f"It is pushing the prediction "
                                f"toward **LEGITIMATE**."

                            )


                except Exception as e:


                    st.error(
                        "❌ SHAP explanation could not be generated."
                    )


                    st.code(
                        str(e)
                    )


            # =================================================
            # ISOLATION FOREST
            # =================================================

            elif model_choice == "Isolation Forest":


                prediction = result.get(

                    "prediction",

                    "UNKNOWN"

                )


                anomaly_score = result.get(

                    "anomaly_score",

                    None

                )


                fraud_probability = result.get(

                    "fraud_probability",

                    None

                )


                col1, col2, col3 = st.columns(3)


                with col1:


                    st.metric(

                        "Detection Result",

                        prediction

                    )


                with col2:


                    if anomaly_score is not None:


                        st.metric(

                            "Anomaly Score",

                            f"{float(anomaly_score):.6f}"

                        )


                    else:


                        st.metric(

                            "Anomaly Score",

                            "N/A"

                        )


                with col3:


                    st.metric(

                        "Selected Model",

                        "Isolation Forest"

                    )
        # =============================================
                    # RESULT
                    # =============================================

                    if prediction.upper() == "FRAUD":


                        st.error(
                            "🚨 ANOMALOUS TRANSACTION DETECTED"
                        )


                        st.write(
                            "Isolation Forest detected unusual "
                            "transaction characteristics."
                        )


                    else:


                        st.success(
                            "✅ NORMAL TRANSACTION PATTERN"
                        )


                        st.write(
                            "This transaction appears similar to "
                            "normal patterns learned by the model."
                        )


                    # =============================================
                    # EXPLANATION
                    # =============================================

                    st.divider()

                    st.subheader(
                        "🔬 Anomaly Detection Explanation"
                    )


                    st.info(
                        "ℹ️ Isolation Forest is an unsupervised "
                        "anomaly detection model."
                    )


                    st.write(
                        "It detects transactions that are "
                        "statistically unusual compared with "
                        "normal transaction patterns."
                    )


                    if anomaly_score is not None:


                        st.write(
                            f"**Anomaly Score:** "
                            f"`{float(anomaly_score):.6f}`"
                        )


                    if fraud_probability is not None:


                        st.write(
                            f"**Relative Risk Score:** "
                            f"`{float(fraud_probability):.2f}%`"
                        )


                    st.warning(
                        "⚠️ An unusual transaction is not always "
                        "confirmed fraud. This model detects anomalies."
                    )


                # =================================================
            # AUTOENCODER
            # =================================================

            elif model_choice == "Autoencoder":


                prediction = result.get(

                    "prediction",

                    "UNKNOWN"

                )


                fraud_probability = result.get(

                    "fraud_probability",

                    0.0

                )


                reconstruction_error = result.get(

                    "reconstruction_error",

                    None

                )


                threshold = result.get(

                    "threshold",

                    None

                )


                col1, col2, col3 = st.columns(3)


                with col1:


                    st.metric(

                        "Risk Score",

                        f"{float(fraud_probability):.2f}%"

                    )


                with col2:


                    if reconstruction_error is not None:


                        st.metric(

                            "Reconstruction Error",

                            f"{float(reconstruction_error):.6f}"

                        )


                    else:


                        st.metric(

                            "Reconstruction Error",

                            "N/A"

                        )


                with col3:


                    st.metric(

                        "Selected Model",

                        "Autoencoder"

                    )


                # =============================================
                # AUTOENCODER RESULT
                # =============================================

                if prediction.upper() == "FRAUD":


                    st.error(
                        "🚨 SUSPICIOUS TRANSACTION DETECTED"
                    )


                    st.write(
                        "The Autoencoder reconstruction error "
                        "is higher than the learned normal "
                        "transaction threshold."
                    )


                else:


                    st.success(
                        "✅ NORMAL TRANSACTION PATTERN"
                    )


                    st.write(
                        "The transaction pattern is similar "
                        "to the normal transactions learned "
                        "by the Autoencoder."
                    )


                # =============================================
                # AUTOENCODER EXPLANATION
                # =============================================

                st.divider()

                st.subheader(
                    "🧠 Autoencoder Explanation"
                )


                st.write(
                    "The Autoencoder learns how normal "
                    "transactions look."
                )


                st.write(
                    "It attempts to reconstruct the input "
                    "transaction. A large reconstruction "
                    "error indicates that the transaction "
                    "is unusual."
                )


                if reconstruction_error is not None:


                    st.write(
                        f"**Reconstruction Error:** "
                        f"`{float(reconstruction_error):.6f}`"
                    )


                if threshold is not None:


                    st.write(
                        f"**Detection Threshold:** "
                        f"`{float(threshold):.6f}`"
                    )


                if (

                    reconstruction_error is not None

                    and threshold is not None

                ):


                    if float(reconstruction_error) > float(threshold):


                        st.warning(
                            "⚠️ Reconstruction error is above "
                            "the normal transaction threshold."
                        )


                    else:


                        st.success(
                            "✅ Reconstruction error is within "
                            "the normal transaction threshold."
                        )


                st.info(
                    "ℹ️ Autoencoder is an unsupervised deep "
                    "learning anomaly detection model."
                )


        # =================================================
        # API ERROR
        # =================================================

        else:


            st.error(
                f"❌ API Error: {response.status_code}"
            )


            st.code(
                response.text
            )
        # =====================================================
    # CONNECTION ERROR
    # =====================================================

    except requests.exceptions.ConnectionError:


        st.error(
            "❌ Cannot connect to FastAPI."
        )


        st.info(
            "Make sure FastAPI is running:\n\n"
            "`py -m uvicorn main:app --reload`"
        )


    # =====================================================
    # TIMEOUT ERROR
    # =====================================================

    except requests.exceptions.Timeout:


        st.error(
            "❌ The API request timed out."
        )


        st.info(
            "Make sure FastAPI is still running correctly."
        )


    # =====================================================
    # OTHER ERROR
    # =====================================================

    except Exception as e:


        st.error(
            "❌ An unexpected error occurred."
        )


        st.code(
            str(e)
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.caption(
    "🏦 Bank Fraud Detection System | "
    "Random Forest + Tuned XGBoost + "
    "Isolation Forest + Autoencoder | "
    "FastAPI + SHAP Explainability"
)