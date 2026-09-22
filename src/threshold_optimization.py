import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")


# ==========================================
# PREPROCESSING
# ==========================================

# Remove unnecessary columns
df = df.drop(
    columns=["nameOrig", "nameDest", "isFlaggedFraud"]
)

# Convert transaction type to numerical values
df = pd.get_dummies(
    df,
    columns=["type"],
    drop_first=True
)


# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop("isFraud", axis=1)
y = df["isFraud"]


# ==========================================
# TRAIN TEST SPLIT
# IMPORTANT: Same random_state as training
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# LOAD TUNED XGBOOST MODEL
# ==========================================

print("\nLoading Tuned XGBoost model...")

model = joblib.load(
    "models/tuned_xgboost_model.pkl"
)

print("Model loaded successfully!")


# ==========================================
# GET FRAUD PROBABILITIES
# ==========================================

print("\nCalculating fraud probabilities...")

y_prob = model.predict_proba(X_test)[:, 1]


# ==========================================
# TEST DIFFERENT THRESHOLDS
# ==========================================

print("\nOptimizing threshold...\n")

results = []

# Test thresholds from 0.01 to 0.50
thresholds = np.arange(0.01, 0.51, 0.01)

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()


    # COST FUNCTION
    # False Negative is much more expensive
    # because missing a fraud is dangerous

    false_negative_cost = fn * 10
    false_positive_cost = fp * 1

    total_cost = (
        false_negative_cost +
        false_positive_cost
    )


    results.append({
        "threshold": threshold,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "total_cost": total_cost
    })


# ==========================================
# FIND BEST THRESHOLD
# ==========================================

results_df = pd.DataFrame(results)

best_result = results_df.loc[
    results_df["total_cost"].idxmin()
]

best_threshold = best_result["threshold"]


print("=" * 50)

print("BEST THRESHOLD:")
print(best_threshold)

print("\nBEST RESULT:")
print(best_result)

print("=" * 50)


# ==========================================
# SAVE RESULTS
# ==========================================

results_df.to_csv(
    "models/threshold_results.csv",
    index=False
)

with open(
    "models/best_threshold.txt",
    "w"
) as file:

    file.write(str(best_threshold))


print("\nThreshold results saved successfully! ✅")


# ==========================================
# ROC-AUC
# ==========================================

auc = roc_auc_score(
    y_test,
    y_prob
)

print(f"\nROC-AUC Score: {auc}")