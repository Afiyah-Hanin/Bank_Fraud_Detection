from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx

from torch_geometric.nn import SAGEConv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

GRAPH_PATH = MODELS_DIR / "fraud_graph.pt"

MODEL_PATH = MODELS_DIR / "gnn_fraud_model.pth"

OUTPUT_PATH = MODELS_DIR / "gnn_fraud_ring_results.csv"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# GRAPH SAGE MODEL
# MUST MATCH train_gnn.py EXACTLY
# ============================================================

class FraudGNN(torch.nn.Module):

    def __init__(self, input_dim):

        super(FraudGNN, self).__init__()

        self.conv1 = SAGEConv(
            input_dim,
            64
        )

        self.conv2 = SAGEConv(
            64,
            32
        )

        self.conv3 = SAGEConv(
            32,
            2
        )


    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.3,
            training=self.training
        )


        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.3,
            training=self.training
        )


        x = self.conv3(
            x,
            edge_index
        )

        return x


# ============================================================
# START
# ============================================================

print("=" * 70)

print("GNN FRAUD RING INTELLIGENCE SYSTEM")

print("=" * 70)


# ============================================================
# LOAD FRAUD GRAPH
# ============================================================

print("\nLoading fraud graph...")

if not GRAPH_PATH.exists():

    print("\nERROR: fraud_graph.pt not found!")

    print(GRAPH_PATH)

    exit()


graph = torch.load(
    GRAPH_PATH,
    map_location=device,
    weights_only=False
)


print("Graph loaded successfully!")


# ============================================================
# EXTRACT GRAPH DATA FROM DICTIONARY
# ============================================================

x = graph["x"]

edge_index = graph["edge_index"]

y = graph["y"]

train_mask = graph["train_mask"]

test_mask = graph["test_mask"]

account_to_id = graph["account_to_id"]


print(f"Nodes: {x.shape[0]}")

print(f"Features: {x.shape[1]}")

print(f"Edges: {edge_index.shape[1]}")


# ============================================================
# LOAD TRAINED GNN MODEL
# ============================================================

print("\nLoading trained GraphSAGE model...")


if not MODEL_PATH.exists():

    print("\nERROR: gnn_fraud_model.pth not found!")

    print(MODEL_PATH)

    exit()


model = FraudGNN(
    input_dim=x.shape[1]
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)


# ============================================================
# HANDLE DIFFERENT MODEL SAVE FORMATS
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        checkpoint = checkpoint["model_state_dict"]


cleaned_checkpoint = {}

for key, value in checkpoint.items():

    new_key = key.replace(
        "module.",
        ""
    )

    cleaned_checkpoint[new_key] = value


model.load_state_dict(
    cleaned_checkpoint
)


model.to(device)

model.eval()


print("GraphSAGE model loaded successfully!")

print(f"Using device: {device}")


# ============================================================
# MOVE GRAPH TO DEVICE
# ============================================================

x = x.to(device)

edge_index = edge_index.to(device)


# ============================================================
# RUN FRAUD PREDICTIONS
# ============================================================

print("\n" + "=" * 70)

print("RUNNING GNN FRAUD PREDICTIONS")

print("=" * 70)


with torch.no_grad():

    output = model(
        x,
        edge_index
    )


    probabilities = torch.softmax(
        output,
        dim=1
    )


    fraud_probabilities = probabilities[
        :,
        1
    ]


    predictions = torch.argmax(
        output,
        dim=1
    )


fraud_probabilities = (
    fraud_probabilities
    .cpu()
    .numpy()
)


predictions = (
    predictions
    .cpu()
    .numpy()
)


print("\nPredictions completed successfully!")


# ============================================================
# FRAUD STATISTICS
# ============================================================

total_nodes = len(predictions)

predicted_fraud = np.sum(
    predictions == 1
)

predicted_legitimate = np.sum(
    predictions == 0
)


print("\n" + "=" * 70)

print("GNN FRAUD DETECTION RESULTS")

print("=" * 70)


print(f"\nTotal Accounts Analyzed: {total_nodes:,}")

print(
    f"Predicted Legitimate Accounts: "
    f"{predicted_legitimate:,}"
)

print(
    f"Predicted Fraud Accounts: "
    f"{predicted_fraud:,}"
)


fraud_percentage = (
    predicted_fraud / total_nodes
) * 100


print(
    f"Predicted Fraud Percentage: "
    f"{fraud_percentage:.4f}%"
)


# ============================================================
# BUILD TRANSACTION NETWORK
# ============================================================

print("\n" + "=" * 70)

print("BUILDING TRANSACTION NETWORK")

print("=" * 70)


print("\nCreating NetworkX graph...")


G = nx.Graph()


edge_array = (
    edge_index
    .cpu()
    .numpy()
)


edges = list(
    zip(
        edge_array[0],
        edge_array[1]
    )
)


print("Adding transaction edges...")


G.add_edges_from(edges)


print(
    f"\nNetwork Nodes: "
    f"{G.number_of_nodes():,}"
)

print(
    f"Network Edges: "
    f"{G.number_of_edges():,}"
)


# ============================================================
# FIND CONNECTED COMPONENTS
# ============================================================

print("\nFinding connected account groups...")


components = list(
    nx.connected_components(G)
)


print(
    f"Connected Groups Found: "
    f"{len(components):,}"
)


# ============================================================
# ANALYZE FRAUD RINGS
# ============================================================

print("\n" + "=" * 70)

print("ANALYZING FRAUD RINGS USING GNN")

print("=" * 70)


fraud_rings = []

ring_id = 1


for component in components:


    component_nodes = list(component)


    # Ignore tiny groups
    if len(component_nodes) < 2:

        continue


    accounts = len(
        component_nodes
    )


    # ========================================================
    # GET GNN FRAUD RESULTS
    # ========================================================

    component_probabilities = (
        fraud_probabilities[
            component_nodes
        ]
    )


    component_predictions = (
        predictions[
            component_nodes
        ]
    )


    fraud_accounts = np.sum(
        component_predictions == 1
    )


    average_fraud_probability = np.mean(
        component_probabilities
    )


    maximum_fraud_probability = np.max(
        component_probabilities
    )


    # ========================================================
    # NETWORK FEATURES
    # ========================================================

    subgraph = G.subgraph(
        component_nodes
    )


    transactions = (
        subgraph.number_of_edges()
    )


    density = nx.density(
        subgraph
    )


    average_connections = (
        2 * transactions / accounts
    )


    # ========================================================
    # FRAUD RING SUSPICION SCORE
    # ========================================================

    # GNN probability = maximum 50 points

    probability_score = (
        average_fraud_probability * 50
    )


    # Fraud account ratio = maximum 30 points

    fraud_ratio = (
        fraud_accounts / accounts
    )


    fraud_account_score = (
        fraud_ratio * 30
    )


    # Network density = maximum 20 points

    density_score = (
        density * 20
    )


    suspicion_score = (

        probability_score

        + fraud_account_score

        + density_score

    )


    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    if suspicion_score >= 70:

        risk_level = "CRITICAL"


    elif suspicion_score >= 50:

        risk_level = "HIGH"


    elif suspicion_score >= 30:

        risk_level = "MEDIUM"


    else:

        risk_level = "LOW"


    # ========================================================
    # STORE RESULTS
    # ========================================================

    fraud_rings.append({

        "ring_id": ring_id,

        "accounts": accounts,

        "transactions": transactions,

        "density": round(
            density,
            4
        ),

        "average_connections": round(
            average_connections,
            2
        ),

        "predicted_fraud_accounts": int(
            fraud_accounts
        ),

        "average_fraud_probability": round(
            average_fraud_probability,
            4
        ),

        "maximum_fraud_probability": round(
            maximum_fraud_probability,
            4
        ),

        "suspicion_score": round(
            suspicion_score,
            2
        ),

        "risk_level": risk_level

    })


    ring_id += 1


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

print("\nCreating fraud ring intelligence report...")


results_df = pd.DataFrame(
    fraud_rings
)


# ============================================================
# SORT RESULTS
# ============================================================

if not results_df.empty:

    results_df = results_df.sort_values(

        by="suspicion_score",

        ascending=False

    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(

    OUTPUT_PATH,

    index=False

)


# ============================================================
# DISPLAY TOP RESULTS
# ============================================================

print("\n" + "=" * 70)

print("TOP 20 SUSPICIOUS FRAUD RINGS")

print("=" * 70)


if not results_df.empty:

    print()

    print(

        results_df.head(20).to_string(

            index=False

        )

    )


else:

    print(
        "\nNo connected fraud rings found."
    )


# ============================================================
# FRAUD RING SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("FRAUD RING INTELLIGENCE SUMMARY")

print("=" * 70)


print(
    f"\nTotal Groups Analyzed: "
    f"{len(results_df):,}"
)


if not results_df.empty:


    critical = np.sum(

        results_df["risk_level"]

        == "CRITICAL"

    )


    high = np.sum(

        results_df["risk_level"]

        == "HIGH"

    )


    medium = np.sum(

        results_df["risk_level"]

        == "MEDIUM"

    )


    low = np.sum(

        results_df["risk_level"]

        == "LOW"

    )


    print(
        f"\n🔴 Critical Risk Rings: "
        f"{critical}"
    )


    print(
        f"🟠 High Risk Rings: "
        f"{high}"
    )


    print(
        f"🟡 Medium Risk Rings: "
        f"{medium}"
    )


    print(
        f"🟢 Low Risk Rings: "
        f"{low}"
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)

print("GNN FRAUD RING ANALYSIS COMPLETED SUCCESSFULLY! 🎉")

print("=" * 70)


print("\nResults saved at:")

print(OUTPUT_PATH)