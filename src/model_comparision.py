import pandas as pd
import matplotlib.pyplot as plt

# Model comparison data
models = {
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    
    "Precision (Fraud)": [
        0.16,
        0.73,
        0.52
    ],
    
    "Recall (Fraud)": [
        0.80,
        0.94,
        0.98
    ],
    
    "F1-Score (Fraud)": [
        0.27,
        0.82,
        0.68
    ],
    
    "ROC-AUC": [
        0.9932,
        0.9987,
        0.9997
    ]
}

df = pd.DataFrame(models)

print("\nMODEL COMPARISON:\n")
print(df)

# Plot ROC-AUC comparison
plt.figure(figsize=(8, 5))

plt.bar(df["Model"], df["ROC-AUC"])

plt.title("Model Comparison - ROC-AUC Score")
plt.xlabel("Models")
plt.ylabel("ROC-AUC Score")

plt.ylim(0.98, 1.0)

plt.show()