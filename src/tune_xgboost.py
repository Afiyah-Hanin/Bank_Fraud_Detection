import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


# =====================================
# 1. LOAD DATASET
# =====================================

file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

print("Loading dataset...")
df = pd.read_csv(file_path)

print("Dataset loaded!")


# =====================================
# 2. SELECT FEATURES
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

# One-hot encoding
df = pd.get_dummies(
    df,
    columns=["type"],
    drop_first=True,
    dtype=int
)


# =====================================
# 3. CREATE A REPRESENTATIVE SUBSET
# =====================================

# Keep ALL fraud transactions
fraud_df = df[df["isFraud"] == 1]

# Randomly sample non-fraud transactions
non_fraud_df = df[df["isFraud"] == 0].sample(
    n=100000,
    random_state=42
)

# Combine them
df_sample = pd.concat(
    [fraud_df, non_fraud_df]
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nTuning dataset shape:", df_sample.shape)
print("\nClass distribution:")
print(df_sample["isFraud"].value_counts())


# =====================================
# 4. FEATURES AND TARGET
# =====================================

X = df_sample.drop("isFraud", axis=1)
y = df_sample["isFraud"]


# =====================================
# 5. TRAIN / VALIDATION SPLIT
# =====================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 6. OPTUNA OBJECTIVE FUNCTION
# =====================================

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 400
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),

        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),

        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    }

    model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, probabilities)

    return score


# =====================================
# 7. RUN OPTUNA
# =====================================

print("\nStarting hyperparameter tuning...")

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=20
)


# =====================================
# 8. DISPLAY RESULTS
# =====================================

print("\n=================================")
print("BEST ROC-AUC SCORE:")
print(study.best_value)

print("\nBEST PARAMETERS:")
print(study.best_params)
print("=================================")