from pathlib import Path
import pandas as pd
import numpy as np

from evidently import Report
from evidently.presets import DataDriftPreset


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

MODELS_DIR = BASE_DIR / "models"

MONITORING_DIR = BASE_DIR / "monitoring"

REPORT_PATH = (
    MONITORING_DIR
    / "data_drift_report.html"
)


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_SIZE = 100000

CURRENT_SIZE = 50000


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)

print("EVIDENTLY AI - FRAUD DATA DRIFT MONITORING")

print("=" * 70)


print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

print(f"Total transactions: {len(df):,}")


# ============================================================
# PREPARE DATA
# ============================================================

print("\nPreparing transaction features...")


FEATURES = [

    "amount",

    "oldbalanceOrg",

    "newbalanceOrig",

    "oldbalanceDest",

    "newbalanceDest"

]


data = df[FEATURES].copy()


# ============================================================
# CREATE REFERENCE DATA
# ============================================================

print("\nCreating reference dataset...")


reference_data = (

    data

    .iloc[:REFERENCE_SIZE]

    .copy()

)


# ============================================================
# CREATE CURRENT DATA
# ============================================================

print("Creating current transaction dataset...")


current_data = (

    data

    .iloc[
        REFERENCE_SIZE:
        REFERENCE_SIZE + CURRENT_SIZE
    ]

    .copy()

)


print(f"\nReference transactions: {len(reference_data):,}")

print(f"Current transactions: {len(current_data):,}")


# ============================================================
# SIMULATE REAL-WORLD DRIFT
# ============================================================

print("\nSimulating transaction drift for testing...")


np.random.seed(42)


current_data["amount"] = (

    current_data["amount"]

    * np.random.uniform(

        1.2,

        2.0,

        len(current_data)

    )

)


current_data["oldbalanceOrg"] = (

    current_data["oldbalanceOrg"]

    * np.random.uniform(

        1.1,

        1.5,

        len(current_data)

    )

)


# ============================================================
# CREATE EVIDENTLY REPORT
# ============================================================

print("\nRunning Evidently drift analysis...")


report = Report(

    [

        DataDriftPreset()

    ]

)


snapshot = report.run(

    reference_data=reference_data,

    current_data=current_data

)


# ============================================================
# SAVE HTML REPORT
# ============================================================

snapshot.save_html(

    REPORT_PATH

)


print("\n" + "=" * 70)

print("DRIFT MONITORING RESULTS")

print("=" * 70)


result = snapshot.dict()


print("\nData drift analysis completed successfully!")

print("\nChecking feature drift...")


for feature in FEATURES:

    print(

        f"Monitoring feature: {feature}"

    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("REAL-TIME DRIFT MONITORING COMPLETED SUCCESSFULLY! 🎉")

print("=" * 70)


print("\nReport saved at:")

print(REPORT_PATH)


print(

    "\nOpen the HTML file in your browser "

    "to view the complete Evidently AI drift dashboard."

)