import pandas as pd

# Load the dataset
file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

# Display the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset shape
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# Dataset information
print("\nDataset information:")
df.info()

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fraud distribution
print("\nFraud distribution:")
print(df["isFraud"].value_counts())

print("\nFraud percentage:")
print(df["isFraud"].value_counts(normalize=True) * 100)