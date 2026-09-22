import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# =====================================
# 1. LOAD DATASET
# =====================================

file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")


# =====================================
# 2. SELECT SAME FEATURES
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
# 4. CREATE FEATURES
# =====================================

X = df.drop("isFraud", axis=1)


# =====================================
# 5. LOAD TUNED XGBOOST MODEL
# =====================================

print("Loading tuned XGBoost model...")

model = joblib.load(
    "models/tuned_xgboost_model.pkl"
)

print("Model loaded successfully! ✅")


# =====================================
# 6. TAKE A SMALL SAMPLE
# =====================================

# SHAP can be computationally expensive,
# so we use only a sample.

X_sample = X.sample(
    n=1000,
    random_state=42
)

print("Creating SHAP explainer...")


# =====================================
# 7. CREATE SHAP EXPLAINER
# =====================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_sample)


# =====================================
# 8. FEATURE IMPORTANCE PLOT
# =====================================

print("Generating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.title("SHAP Feature Importance - Fraud Detection")

plt.tight_layout()

plt.savefig(
    "reports/shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("SHAP summary plot saved successfully! ✅")