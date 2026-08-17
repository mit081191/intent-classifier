from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
# 2. Load drift dataset
# ---------------------------------------------------------

drift_df = pd.read_csv(DRIFT_FILE)

print("Drift dataset:", drift_df.shape)
print(
    "Unique intents:",
    drift_df["intent"].nunique(),
)


# ---------------------------------------------------------
# 3. Validate the controlled drift dataset
# ---------------------------------------------------------

required_columns = {
    "utterance",
    "intent",
}

missing_columns = (
    required_columns
    - set(drift_df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Check whether every intent has the same number
# of controlled drift examples.
intent_counts = (
    drift_df["intent"]
    .value_counts()
    .sort_index()
)

print("\nExamples per intent:")
print(intent_counts)


# ---------------------------------------------------------
# 4. Load selected production model
# ---------------------------------------------------------

# This is the exact TF-IDF + Linear SVM pipeline
# selected during Week 2.
model = joblib.load(
    MODEL_FILE
)


# ---------------------------------------------------------
# 5. Run predictions on drift data
# ---------------------------------------------------------

X_drift = drift_df["utterance"]
y_drift = drift_df["intent"]

y_pred = model.predict(
    X_drift
)


# ---------------------------------------------------------
# 6. Calculate performance on drift data
# ---------------------------------------------------------

drift_accuracy = accuracy_score(
    y_drift,
    y_pred,
)

drift_macro_f1 = f1_score(
    y_drift,
    y_pred,
    average="macro",
)


# ---------------------------------------------------------
# 7. Compare against final test baseline
# ---------------------------------------------------------

# These values came from our untouched final test evaluation.
BASELINE_TEST_ACCURACY = 0.9976
BASELINE_TEST_MACRO_F1 = 0.9975

accuracy_drop = (
    BASELINE_TEST_ACCURACY
    - drift_accuracy
)

macro_f1_drop = (
    BASELINE_TEST_MACRO_F1
    - drift_macro_f1
)


print("\n" + "=" * 70)
print("DRIFT DATASET PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy : {drift_accuracy:.4f}"
)

print(
    f"Macro F1 : {drift_macro_f1:.4f}"
)


print("\n" + "=" * 70)
print("PERFORMANCE CHANGE FROM ORIGINAL TEST SET")
print("=" * 70)

print(
    f"Original Test Accuracy : "
    f"{BASELINE_TEST_ACCURACY:.4f}"
)

print(
    f"Drift Accuracy         : "
    f"{drift_accuracy:.4f}"
)

print(
    f"Accuracy Drop          : "
    f"{accuracy_drop:.4f}"
)

print()

print(
    f"Original Test Macro F1 : "
    f"{BASELINE_TEST_MACRO_F1:.4f}"
)

print(
    f"Drift Macro F1         : "
    f"{drift_macro_f1:.4f}"
)

print(
    f"Macro F1 Drop          : "
    f"{macro_f1_drop:.4f}"
)


# ---------------------------------------------------------
# 8. Detailed classification report
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DRIFT CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_drift,
        y_pred,
        digits=4,
        zero_division=0,
    )
)


# ---------------------------------------------------------
# 9. Error analysis
# ---------------------------------------------------------

error_analysis = drift_df.copy()

error_analysis[
    "predicted_intent"
] = y_pred

misclassified = error_analysis[
    error_analysis["intent"]
    != error_analysis["predicted_intent"]
]


print("\n" + "=" * 70)
print("MISCLASSIFIED DRIFT EXAMPLES")
print("=" * 70)

print(
    f"Total misclassified examples: "
    f"{len(misclassified)}"
)

pd.set_option(
    "display.max_colwidth",
    None,
)

if len(misclassified) > 0:

    print(
        misclassified[
            [
                "utterance",
                "intent",
                "predicted_intent",
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No misclassified drift examples."
    )