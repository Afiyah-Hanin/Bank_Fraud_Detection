import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data/raw/PS_20174392719_1491204439457_log.csv"
df = pd.read_csv(file_path)

# -------------------------------
# 1. Transaction type distribution
# -------------------------------

print("\nTransaction Types:")
print(df["type"].value_counts())

print("\nFraud by Transaction Type:")
print(pd.crosstab(df["type"], df["isFraud"]))


# -------------------------------
# 2. Fraud rate by transaction type
# -------------------------------

fraud_rate = df.groupby("type")["isFraud"].mean() * 100

print("\nFraud Rate by Transaction Type (%):")
print(fraud_rate)


# -------------------------------
# 3. Basic statistics
# -------------------------------

print("\nTransaction Amount Statistics:")
print(df["amount"].describe())


# -------------------------------
# 4. Visualizations
# -------------------------------

# Transaction type distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="type")
plt.title("Transaction Type Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Fraud rate by transaction type
plt.figure(figsize=(8, 5))
fraud_rate.sort_values().plot(kind="bar")
plt.title("Fraud Rate by Transaction Type")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Fraud vs Non-Fraud
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="isFraud")
plt.title("Fraud vs Non-Fraud Transactions")
plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
plt.tight_layout()
plt.show()