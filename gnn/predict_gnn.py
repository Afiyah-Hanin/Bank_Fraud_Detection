import pandas as pd
import numpy as np
import torch

from torch_geometric.data import Data


# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = "../data/PS_20174392719_1491204439457_log.csv"

MAX_ROWS = 100000


# =========================================================
# LOAD DATASET
# =========================================================

print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    nrows=MAX_ROWS
)

print("Dataset loaded!")

print("Shape:", df.shape)


# =========================================================
# REMOVE TRANSACTION TYPES NOT NEEDED
# =========================================================

df = df[
    df["type"].isin(
        [
            "TRANSFER",
            "CASH_OUT",
            "PAYMENT",
            "DEBIT"
        ]
    )
].copy()


# =========================================================
# CREATE ACCOUNT LIST
# =========================================================

all_accounts = pd.concat(
    [
        df["nameOrig"],
        df["nameDest"]
    ]
).unique()


account_to_id = {
    account: index
    for index, account in enumerate(all_accounts)
}


print("Total Accounts:", len(account_to_id))


# =========================================================
# CREATE EDGES
# =========================================================

source_nodes = df["nameOrig"].map(
    account_to_id
).values


destination_nodes = df["nameDest"].map(
    account_to_id
).values


edge_index = torch.tensor(
    [
        source_nodes,
        destination_nodes
    ],
    dtype=torch.long
)


print("Total Transactions:", edge_index.shape[1])


# =========================================================
# CREATE NODE FEATURES
# =========================================================

num_nodes = len(account_to_id)


# Basic account features

node_features = np.zeros(
    (num_nodes, 4),
    dtype=np.float32
)


# Feature 1:
# Number of outgoing transactions

outgoing_count = df.groupby(
    "nameOrig"
).size()


# Feature 2:
# Number of incoming transactions

incoming_count = df.groupby(
    "nameDest"
).size()


# Feature 3:
# Total money sent

total_sent = df.groupby(
    "nameOrig"
)["amount"].sum()


# Feature 4:
# Total money received

total_received = df.groupby(
    "nameDest"
)["amount"].sum()


for account, node_id in account_to_id.items():

    node_features[node_id][0] = outgoing_count.get(
        account,
        0
    )

    node_features[node_id][1] = incoming_count.get(
        account,
        0
    )

    node_features[node_id][2] = total_sent.get(
        account,
        0
    )

    node_features[node_id][3] = total_received.get(
        account,
        0
    )


# =========================================================
# NORMALIZE FEATURES
# =========================================================

node_features = np.log1p(
    node_features
)


x = torch.tensor(
    node_features,
    dtype=torch.float
)


# =========================================================
# CREATE FRAUD LABELS FOR ACCOUNTS
# =========================================================

fraud_accounts = set(
    df.loc[
        df["isFraud"] == 1,
        "nameOrig"
    ]
)


fraud_accounts.update(

    df.loc[
        df["isFraud"] == 1,
        "nameDest"
    ]

)


node_labels = np.zeros(
    num_nodes,
    dtype=np.int64
)


for account in fraud_accounts:

    node_id = account_to_id[account]

    node_labels[node_id] = 1


y = torch.tensor(
    node_labels,
    dtype=torch.long
)


# =========================================================
# CREATE TRAIN / TEST MASKS
# =========================================================

indices = np.random.permutation(
    num_nodes
)


train_size = int(
    0.8 * num_nodes
)


train_indices = indices[:train_size]

test_indices = indices[train_size:]


train_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

test_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)


train_mask[train_indices] = True

test_mask[test_indices] = True


# =========================================================
# CREATE GRAPH DATA
# =========================================================

graph_data = Data(

    x=x,

    edge_index=edge_index,

    y=y,

    train_mask=train_mask,

    test_mask=test_mask

)


# =========================================================
# SAVE GRAPH
# =========================================================

torch.save(
    graph_data,
    "../models/fraud_graph.pt"
)


# =========================================================
# SAVE ACCOUNT MAPPING
# =========================================================

import joblib

joblib.dump(
    account_to_id,
    "../models/account_to_id.pkl"
)


print("\nGraph created successfully!")

print(graph_data)