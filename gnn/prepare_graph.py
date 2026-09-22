from pathlib import Path

import pandas as pd
import numpy as np
import torch


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"

# Automatically create models folder if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = MODELS_DIR / "fraud_graph.pt"


# =========================================================
# SETTINGS
# =========================================================

# IMPORTANT:
# Your full dataset has millions of transactions and accounts.
# Creating the complete graph requires a huge amount of RAM.
#
# Start with a sample first.
MAX_ROWS = 200000


# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

print(f"Dataset path: {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )

print("\nReading dataset...")

df = pd.read_csv(
    DATA_PATH,
    nrows=MAX_ROWS
)

print("Dataset loaded successfully!")

print(f"Dataset shape: {df.shape}")


# =========================================================
# DISPLAY COLUMNS
# =========================================================

print("\nDataset columns:")

print(df.columns.tolist())


# =========================================================
# REMOVE TRANSACTION TYPES NOT NEEDED
# =========================================================

# PaySim transaction types:
#
# CASH_OUT
# PAYMENT
# CASH_IN
# TRANSFER
# DEBIT

print("\nFiltering transactions...")

df = df[
    df["type"].isin(
        [
            "TRANSFER",
            "CASH_OUT",
            "PAYMENT",
            "DEBIT",
            "CASH_IN"
        ]
    )
].copy()

print(f"Transactions after filtering: {len(df)}")


# =========================================================
# CREATE ACCOUNT IDs
# =========================================================

print("\nCreating account nodes...")

all_accounts = pd.concat(
    [
        df["nameOrig"],
        df["nameDest"]
    ]
).unique()

print(f"Total Accounts: {len(all_accounts)}")


# =========================================================
# MAP ACCOUNT NAMES TO NODE IDs
# =========================================================

account_to_id = {
    account: index
    for index, account in enumerate(all_accounts)
}


# =========================================================
# CREATE SOURCE AND DESTINATION NODES
# =========================================================

print("\nCreating graph connections...")

source_nodes = (
    df["nameOrig"]
    .map(account_to_id)
    .to_numpy()
)

destination_nodes = (
    df["nameDest"]
    .map(account_to_id)
    .to_numpy()
)


# =========================================================
# CREATE EDGE INDEX
# =========================================================

print("\nCreating edge index...")

edge_index_array = np.vstack(
    [
        source_nodes,
        destination_nodes
    ]
)

edge_index = torch.tensor(
    edge_index_array,
    dtype=torch.long
)

print(f"Total Transactions: {edge_index.shape[1]}")


# =========================================================
# CREATE NODE FEATURES
# =========================================================

print("\nCreating node features...")

num_nodes = len(all_accounts)


# ---------------------------------------------------------
# Features:
#
# 0 = total outgoing transactions
# 1 = total incoming transactions
# 2 = total outgoing amount
# 3 = total incoming amount
# 4 = fraud involvement count
# ---------------------------------------------------------


node_features = np.zeros(
    (
        num_nodes,
        5
    ),
    dtype=np.float32
)


# =========================================================
# CALCULATE TRANSACTION STATISTICS
# =========================================================

print("Calculating account statistics...")


# ---------------------------------------------------------
# OUTGOING TRANSACTION COUNT
# ---------------------------------------------------------

np.add.at(
    node_features[:, 0],
    source_nodes,
    1
)


# ---------------------------------------------------------
# INCOMING TRANSACTION COUNT
# ---------------------------------------------------------

np.add.at(
    node_features[:, 1],
    destination_nodes,
    1
)


# =========================================================
# TRANSACTION AMOUNTS
# =========================================================

amounts = df["amount"].to_numpy(
    dtype=np.float32
)


# ---------------------------------------------------------
# OUTGOING AMOUNT
# ---------------------------------------------------------

np.add.at(
    node_features[:, 2],
    source_nodes,
    amounts
)


# ---------------------------------------------------------
# INCOMING AMOUNT
# ---------------------------------------------------------

np.add.at(
    node_features[:, 3],
    destination_nodes,
    amounts
)


# =========================================================
# FRAUD INFORMATION
# =========================================================

print("Adding fraud labels...")


fraud_labels = df["isFraud"].to_numpy(
    dtype=np.float32
)


# Add fraud involvement to sender accounts

np.add.at(
    node_features[:, 4],
    source_nodes,
    fraud_labels
)


# Add fraud involvement to receiver accounts

np.add.at(
    node_features[:, 4],
    destination_nodes,
    fraud_labels
)


# =========================================================
# NORMALIZE NODE FEATURES
# =========================================================

print("Normalizing node features...")


# Log transformation for large transaction amounts

node_features[:, 0] = np.log1p(
    node_features[:, 0]
)

node_features[:, 1] = np.log1p(
    node_features[:, 1]
)

node_features[:, 2] = np.log1p(
    node_features[:, 2]
)

node_features[:, 3] = np.log1p(
    node_features[:, 3]
)

node_features[:, 4] = np.log1p(
    node_features[:, 4]
)


# =========================================================
# CONVERT FEATURES TO PYTORCH
# =========================================================

x = torch.tensor(
    node_features,
    dtype=torch.float
)


# =========================================================
# CREATE NODE LABELS
# =========================================================

print("Creating fraud labels for accounts...")


# A node is labeled fraud if it participated
# in one or more fraudulent transactions

node_labels = (
    node_features[:, 4] > 0
).astype(
    np.int64
)


y = torch.tensor(
    node_labels,
    dtype=torch.long
)


# =========================================================
# CREATE TRAIN / TEST MASKS
# =========================================================

print("Creating train and test masks...")


torch.manual_seed(42)


indices = torch.randperm(
    num_nodes
)


train_size = int(
    0.8 * num_nodes
)


train_indices = indices[
    :train_size
]

test_indices = indices[
    train_size:
]


train_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

test_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)


train_mask[
    train_indices
] = True


test_mask[
    test_indices
] = True


# =========================================================
# CREATE GRAPH DATA
# =========================================================

print("\nCreating graph object...")


graph_data = {

    "x": x,

    "edge_index": edge_index,

    "y": y,

    "train_mask": train_mask,

    "test_mask": test_mask,

    "account_to_id": account_to_id

}


# =========================================================
# SAVE GRAPH
# =========================================================

print("\n" + "=" * 60)
print("SAVING GRAPH")
print("=" * 60)


print(f"Saving to:\n{OUTPUT_PATH}")


torch.save(
    graph_data,
    OUTPUT_PATH
)


# =========================================================
# FINAL INFORMATION
# =========================================================

print("\n" + "=" * 60)
print("GRAPH CREATED SUCCESSFULLY!")
print("=" * 60)

print(f"\nNumber of Nodes: {num_nodes}")

print(
    f"Number of Edges: "
    f"{edge_index.shape[1]}"
)

print(
    f"Node Feature Shape: "
    f"{x.shape}"
)

print(
    f"Fraud Nodes: "
    f"{int(y.sum())}"
)

print(
    f"Legitimate Nodes: "
    f"{int((y == 0).sum())}"
)

print(
    f"\nGraph saved successfully at:\n"
    f"{OUTPUT_PATH}"
)

print("\nNEXT STEP:")
print(
    "Train the Graph Neural Network using "
    "gnn/train_gnn.py"
)