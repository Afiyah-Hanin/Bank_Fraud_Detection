import torch
import pandas as pd
import networkx as nx
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

GRAPH_PATH = BASE_DIR / "models" / "fraud_graph.pt"
OUTPUT_PATH = BASE_DIR / "models" / "fraud_rings.csv"


print("=" * 65)
print("FRAUD RING ANALYSIS")
print("=" * 65)


# ============================================================
# LOAD GRAPH
# ============================================================

print("\nLoading fraud graph...")

graph_data = torch.load(
    GRAPH_PATH,
    weights_only=False
)

print("Graph loaded successfully!")

print(f"\nNodes: {graph_data['x'].shape[0]}")
print(f"Edges: {graph_data['edge_index'].shape[1]}")


# ============================================================
# CREATE NETWORK GRAPH
# ============================================================

print("\nCreating transaction network...")

edge_index = graph_data["edge_index"].cpu().numpy()

G = nx.Graph()

edges = edge_index.T.tolist()

G.add_edges_from(edges)

print(f"Network Nodes: {G.number_of_nodes()}")
print(f"Network Edges: {G.number_of_edges()}")


# ============================================================
# FIND CONNECTED COMPONENTS
# ============================================================

print("\nFinding connected account groups...")

components = list(nx.connected_components(G))

print(f"Total Connected Components: {len(components)}")


# ============================================================
# ANALYZE LARGE GROUPS
# ============================================================

suspicious_rings = []

print("\nAnalyzing potential fraud rings...")


for ring_id, component in enumerate(components):

    ring_size = len(component)

    # Ignore very small groups
    if ring_size < 3:
        continue

    subgraph = G.subgraph(component)

    num_edges = subgraph.number_of_edges()

    # Graph density
    if ring_size > 1:

        density = nx.density(subgraph)

    else:

        density = 0


    # Average degree
    degrees = [degree for node, degree in subgraph.degree()]

    avg_degree = sum(degrees) / len(degrees)


    # Suspicion score
    suspicion_score = (
        ring_size * 0.2
        + density * 50
        + avg_degree * 2
    )


    suspicious_rings.append({

        "ring_id": ring_id,

        "accounts": ring_size,

        "transactions": num_edges,

        "density": round(density, 4),

        "average_connections": round(avg_degree, 2),

        "suspicion_score": round(suspicion_score, 2)

    })


# ============================================================
# CREATE RESULTS
# ============================================================

results = pd.DataFrame(suspicious_rings)


if len(results) == 0:

    print("\nNo fraud rings detected.")

else:

    results = results.sort_values(
        by="suspicion_score",
        ascending=False
    )


    print("\n" + "=" * 65)
    print("TOP SUSPICIOUS FRAUD RINGS")
    print("=" * 65)

    print(results.head(20).to_string(index=False))


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print("\n" + "=" * 65)
    print("FRAUD RING ANALYSIS COMPLETED! 🎉")
    print("=" * 65)

    print(f"\nResults saved at:")

    print(OUTPUT_PATH)

    print(f"\nTotal suspicious groups analyzed: {len(results)}")