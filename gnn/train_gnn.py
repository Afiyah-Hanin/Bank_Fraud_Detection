from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRAPH_PATH = PROJECT_ROOT / "models" / "fraud_graph.pt"

MODEL_PATH = PROJECT_ROOT / "models" / "gnn_fraud_model.pth"


# =========================================================
# SETTINGS
# =========================================================

RANDOM_SEED = 42

HIDDEN_CHANNELS_1 = 64
HIDDEN_CHANNELS_2 = 32

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

MAX_EPOCHS = 100

PATIENCE = 15


# =========================================================
# RANDOM SEED
# =========================================================

torch.manual_seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)


# =========================================================
# LOAD GRAPH
# =========================================================

print("=" * 65)
print("LOADING FRAUD GRAPH")
print("=" * 65)

print(f"\nGraph path:\n{GRAPH_PATH}")

if not GRAPH_PATH.exists():

    raise FileNotFoundError(
        f"\nGraph file not found:\n{GRAPH_PATH}\n\n"
        "Run gnn/prepare_graph.py first."
    )


graph = torch.load(
    GRAPH_PATH,
    weights_only=False
)


x = graph["x"]

edge_index = graph["edge_index"]

y = graph["y"]

print("\nGraph loaded successfully!")

print(f"Nodes: {x.shape[0]}")

print(f"Features per node: {x.shape[1]}")

print(f"Edges: {edge_index.shape[1]}")


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing device: {device}")


# =========================================================
# FRAUD STATISTICS
# =========================================================

print("\n" + "=" * 65)
print("DATASET STATISTICS")
print("=" * 65)

total_nodes = len(y)

fraud_nodes = int((y == 1).sum())

legitimate_nodes = int((y == 0).sum())


print(f"\nTotal Nodes: {total_nodes:,}")

print(f"Legitimate Nodes: {legitimate_nodes:,}")

print(f"Fraud Nodes: {fraud_nodes:,}")

print(
    f"Fraud Percentage: "
    f"{(fraud_nodes / total_nodes) * 100:.4f}%"
)


# =========================================================
# CREATE TRAIN / VALIDATION / TEST SPLIT
# =========================================================

print("\n" + "=" * 65)
print("CREATING TRAIN / VALIDATION / TEST SPLIT")
print("=" * 65)


# ---------------------------------------------------------
# We split fraud and legitimate nodes separately.
#
# This ensures every dataset contains fraud examples.
# ---------------------------------------------------------


fraud_indices = torch.where(
    y == 1
)[0]


legitimate_indices = torch.where(
    y == 0
)[0]


# Shuffle separately

fraud_indices = fraud_indices[
    torch.randperm(
        len(fraud_indices)
    )
]


legitimate_indices = legitimate_indices[
    torch.randperm(
        len(legitimate_indices)
    )
]


# =========================================================
# SPLIT FUNCTION
# =========================================================

def split_indices(indices):

    total = len(indices)

    train_end = int(
        total * 0.70
    )

    val_end = int(
        total * 0.85
    )


    train = indices[
        :train_end
    ]

    validation = indices[
        train_end:val_end
    ]

    test = indices[
        val_end:
    ]


    return train, validation, test


# =========================================================
# SPLIT FRAUD NODES
# =========================================================

fraud_train, fraud_val, fraud_test = split_indices(
    fraud_indices
)


# =========================================================
# SPLIT LEGITIMATE NODES
# =========================================================

legit_train, legit_val, legit_test = split_indices(
    legitimate_indices
)


# =========================================================
# CREATE MASKS
# =========================================================

train_mask = torch.zeros(
    total_nodes,
    dtype=torch.bool
)

val_mask = torch.zeros(
    total_nodes,
    dtype=torch.bool
)

test_mask = torch.zeros(
    total_nodes,
    dtype=torch.bool
)


# =========================================================
# TRAIN MASK
# =========================================================

train_mask[
    fraud_train
] = True

train_mask[
    legit_train
] = True


# =========================================================
# VALIDATION MASK
# =========================================================

val_mask[
    fraud_val
] = True

val_mask[
    legit_val
] = True


# =========================================================
# TEST MASK
# =========================================================

test_mask[
    fraud_test
] = True

test_mask[
    legit_test
] = True


# =========================================================
# DISPLAY SPLIT STATISTICS
# =========================================================

def show_split(name, mask):

    fraud_count = int(
        (y[mask] == 1).sum()
    )

    legit_count = int(
        (y[mask] == 0).sum()
    )


    print(f"\n{name}")

    print(f"Total: {int(mask.sum()):,}")

    print(f"Legitimate: {legit_count:,}")

    print(f"Fraud: {fraud_count:,}")


show_split(
    "TRAIN SET",
    train_mask
)

show_split(
    "VALIDATION SET",
    val_mask
)

show_split(
    "TEST SET",
    test_mask
)


# =========================================================
# MOVE DATA TO DEVICE
# =========================================================

x = x.to(device)

edge_index = edge_index.to(device)

y = y.to(device)

train_mask = train_mask.to(device)

val_mask = val_mask.to(device)

test_mask = test_mask.to(device)


# =========================================================
# GNN MODEL
# =========================================================

class FraudGNN(torch.nn.Module):


    def __init__(
        self,
        input_channels
    ):

        super().__init__()


        # GraphSAGE Layer 1

        self.conv1 = SAGEConv(
            input_channels,
            HIDDEN_CHANNELS_1
        )


        # GraphSAGE Layer 2

        self.conv2 = SAGEConv(
            HIDDEN_CHANNELS_1,
            HIDDEN_CHANNELS_2
        )


        # Output Layer

        self.conv3 = SAGEConv(
            HIDDEN_CHANNELS_2,
            2
        )


    def forward(
        self,
        x,
        edge_index
    ):


        # Layer 1

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.30,
            training=self.training
        )


        # Layer 2

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.30,
            training=self.training
        )


        # Output

        x = self.conv3(
            x,
            edge_index
        )


        return x


# =========================================================
# CREATE MODEL
# =========================================================

model = FraudGNN(
    input_channels=x.shape[1]
).to(device)


# =========================================================
# CLASS IMBALANCE HANDLING
# =========================================================

print("\n" + "=" * 65)
print("CALCULATING CLASS WEIGHTS")
print("=" * 65)


train_labels = y[
    train_mask
]


class_counts = torch.bincount(
    train_labels
)


print(
    f"\nLegitimate training nodes: "
    f"{class_counts[0].item():,}"
)

print(
    f"Fraud training nodes: "
    f"{class_counts[1].item():,}"
)


# ---------------------------------------------------------
# Balanced class weights
# ---------------------------------------------------------

total_training_samples = class_counts.sum()


class_weights = (
    total_training_samples
    /
    (
        2
        *
        class_counts.float()
    )
)


class_weights = class_weights.to(device)


print(
    f"\nClass weights:\n"
    f"{class_weights}"
)


# =========================================================
# LOSS FUNCTION
# =========================================================

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =========================================================
# TRAIN FUNCTION
# =========================================================

def train():

    model.train()

    optimizer.zero_grad()


    output = model(
        x,
        edge_index
    )


    loss = criterion(
        output[train_mask],
        y[train_mask]
    )


    loss.backward()


    optimizer.step()


    return loss.item()


# =========================================================
# EVALUATION FUNCTION
# =========================================================

@torch.no_grad()

def evaluate(mask):

    model.eval()


    output = model(
        x,
        edge_index
    )


    probabilities = torch.softmax(
        output,
        dim=1
    )[:, 1]


    predictions = (
        probabilities >= 0.50
    ).long()


    true_labels = (
        y[mask]
        .cpu()
        .numpy()
    )


    predicted_labels = (
        predictions[mask]
        .cpu()
        .numpy()
    )


    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )


    precision = precision_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )


    recall = recall_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )


    f1 = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )


    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1

    }


# =========================================================
# TRAINING
# =========================================================

print("\n" + "=" * 65)
print("STARTING IMPROVED GNN TRAINING")
print("=" * 65)


best_f1 = 0

epochs_without_improvement = 0


for epoch in range(
    1,
    MAX_EPOCHS + 1
):


    # =====================================================
    # TRAIN
    # =====================================================

    loss = train()


    # =====================================================
    # VALIDATION
    # =====================================================

    validation_metrics = evaluate(
        val_mask
    )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print(

        f"\nEpoch {epoch:03d}/{MAX_EPOCHS}"

    )

    print(
        f"Loss: {loss:.6f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Validation Precision: "
        f"{validation_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Validation Recall: "
        f"{validation_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"Validation F1 Score: "
        f"{validation_metrics['f1'] * 100:.2f}%"
    )


    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    current_f1 = validation_metrics[
        "f1"
    ]


    if current_f1 > best_f1:


        best_f1 = current_f1

        epochs_without_improvement = 0


        torch.save(

            {

                "model_state_dict":
                    model.state_dict(),

                "input_features":
                    x.shape[1],

                "hidden_channels_1":
                    HIDDEN_CHANNELS_1,

                "hidden_channels_2":
                    HIDDEN_CHANNELS_2,

                "output_classes":
                    2,

                "best_validation_f1":
                    best_f1

            },

            MODEL_PATH

        )


        print(
            "🔥 New best model saved!"
        )


    else:

        epochs_without_improvement += 1


    # =====================================================
    # EARLY STOPPING
    # =====================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):


        print(
            "\n⏹️ Early stopping activated."
        )

        print(
            f"No F1 improvement for "
            f"{PATIENCE} epochs."
        )

        break


# =========================================================
# LOAD BEST MODEL
# =========================================================

print("\n" + "=" * 65)
print("LOADING BEST MODEL")
print("=" * 65)


checkpoint = torch.load(
    MODEL_PATH,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


# =========================================================
# FINAL TEST EVALUATION
# =========================================================

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)


test_metrics = evaluate(
    test_mask
)


print(
    f"\nAccuracy: "
    f"{test_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{test_metrics['precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{test_metrics['recall'] * 100:.2f}%"
)

print(
    f"F1 Score: "
    f"{test_metrics['f1'] * 100:.2f}%"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

with torch.no_grad():

    output = model(
        x,
        edge_index
    )


    probabilities = torch.softmax(
        output,
        dim=1
    )[:, 1]


    predictions = (
        probabilities >= 0.50
    ).long()


true_labels = (
    y[test_mask]
    .cpu()
    .numpy()
)


predicted_labels = (
    predictions[test_mask]
    .cpu()
    .numpy()
)


cm = confusion_matrix(
    true_labels,
    predicted_labels
)


tn, fp, fn, tp = cm.ravel()


print("\nConfusion Matrix:")

print(f"True Negative:  {tn:,}")

print(f"False Positive: {fp:,}")

print(f"False Negative: {fn:,}")

print(f"True Positive:  {tp:,}")


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 65)
print("GNN TRAINING COMPLETED SUCCESSFULLY! 🎉")
print("=" * 65)


print(
    f"\nBest Validation F1 Score: "
    f"{checkpoint['best_validation_f1'] * 100:.2f}%"
)


print(
    f"\nModel saved at:\n"
    f"{MODEL_PATH}"
)


print("\nNEXT STEP:")

print(
    "Create fraud ring analysis and "
    "then integrate the GNN into FastAPI."
)