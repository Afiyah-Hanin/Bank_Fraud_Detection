import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# -----------------------------------
# 1. Load Dataset
# -----------------------------------

file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# -----------------------------------
# 2. Select Features
# -----------------------------------

features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

target = "isFraud"

df_model = df[features + [target]].copy()


# -----------------------------------
# 3. Encode Transaction Type
# -----------------------------------

df_encoded = pd.get_dummies(
    df_model,
    columns=["type"],
    drop_first=True,
    dtype=int
)

print("\nShape after encoding:", df_encoded.shape)


# -----------------------------------
# 4. Separate Features and Target
# -----------------------------------

X = df_encoded.drop("isFraud", axis=1)
y = df_encoded["isFraud"]


# -----------------------------------
# 5. Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nBefore SMOTE:")
print(y_train.value_counts())


# -----------------------------------
# 6. Apply SMOTE ONLY to Training Data
# -----------------------------------

smote = SMOTE(
    sampling_strategy=0.1,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

print("\nTraining shape after SMOTE:")
print(X_train_smote.shape)

print("\nPreprocessing completed successfully!")