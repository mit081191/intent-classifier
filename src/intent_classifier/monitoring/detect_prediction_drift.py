from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Customer_Service_Validation_Dataset.csv"
)

DRIFT_FILE = (
    PROJECT_ROOT
    / "data"
    / "drift"
    / "Customer_Service_Drift_Dataset.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "linear_svm_pipeline.joblib"
)


# ---------------------------------------------------------
# 2. Load data and production model
# ---------------------------------------------------------

validation_df = pd.read_csv(
    VALIDATION_FILE
)

drift_df = pd.read_csv(
    DRIFT_FILE
)

model = joblib.load(
    MODEL_FILE
)


# ---------------------------------------------------------
# 3. Generate model predictions
# ---------------------------------------------------------

# IMPORTANT:
# We intentionally compare MODEL PREDICTIONS rather than
# using the true intent labels.
#
# This allows the same idea to work on production traffic
# where ground-truth labels may not yet be available.

validation_predictions = model.predict(
    validation_df["utterance"]
)

drift_predictions = model.predict(
    drift_df["utterance"]
)


# ---------------------------------------------------------
# 4. Get complete intent list
# ---------------------------------------------------------

# LinearSVC stores all trained classes here.
all_intents = model.named_steps[
    "classifier"
].classes_


# ---------------------------------------------------------
# 5. Calculate prediction distributions
# ---------------------------------------------------------

def calculate_distribution(
    predictions,
    intents,
):
    """
    Return the fraction of predictions assigned
    to every possible intent.
    """

    counts = (
        pd.Series(predictions)
        .value_counts(normalize=True)
        .reindex(
            intents,
            fill_value=0.0,
        )
    )

    return counts


validation_distribution = calculate_distribution(
    validation_predictions,
    all_intents,
)

drift_distribution = calculate_distribution(
    drift_predictions,
    all_intents,
)


# ---------------------------------------------------------
# 6. Total Variation Distance
# ---------------------------------------------------------

# Total Variation Distance compares two categorical
# probability distributions.
#
# TVD = 0:
# distributions are identical.
#
# TVD = 1:
# distributions are completely different.
#
# Formula:
#
# 0.5 * SUM(|P_i - Q_i|)

tvd = 0.5 * np.abs(
    validation_distribution.values
    - drift_distribution.values
).sum()


# ---------------------------------------------------------
# 7. Per-intent distribution change
# ---------------------------------------------------------

comparison_df = pd.DataFrame(
    {
        "intent": all_intents,
        "reference_rate":
            validation_distribution.values,
        "current_rate":
            drift_distribution.values,
    }
)

comparison_df[
    "absolute_change"
] = np.abs(
    comparison_df["current_rate"]
    - comparison_df["reference_rate"]
)


comparison_df = comparison_df.sort_values(
    "absolute_change",
    ascending=False,
)


# ---------------------------------------------------------
# 8. Determine monitoring status
# ---------------------------------------------------------

# These are operational alert bands for our project.
#
# They are NOT universal ML thresholds.
#
# TVD <= 0.10 -> LOW
# TVD <= 0.20 -> MODERATE
# TVD > 0.20  -> HIGH
#
# In a real production system these would be calibrated
# from historical stable traffic.

if tvd <= 0.10:
    prediction_drift_status = "LOW"

elif tvd <= 0.20:
    prediction_drift_status = "MODERATE"

else:
    prediction_drift_status = "HIGH"


# ---------------------------------------------------------
# 9. Print results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION DRIFT")
print("=" * 70)

print(
    f"Total Variation Distance : "
    f"{tvd:.4f}"
)

print(
    f"Prediction Drift Status  : "
    f"{prediction_drift_status}"
)


print("\n" + "=" * 70)
print("LARGEST PREDICTION DISTRIBUTION CHANGES")
print("=" * 70)

display_df = comparison_df.copy()

display_df[
    "reference_rate"
] *= 100

display_df[
    "current_rate"
] *= 100

display_df[
    "absolute_change"
] *= 100


print(
    display_df.head(10).to_string(
        index=False,
        formatters={
            "reference_rate":
                lambda x: f"{x:.2f}%",
            "current_rate":
                lambda x: f"{x:.2f}%",
            "absolute_change":
                lambda x: f"{x:.2f}%",
        },
    )
)