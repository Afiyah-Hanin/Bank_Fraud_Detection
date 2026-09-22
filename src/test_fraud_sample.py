import pandas as pd

file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

# Get one fraudulent transaction
fraud_transaction = df[df["isFraud"] == 0].head(1)

print(fraud_transaction.T)