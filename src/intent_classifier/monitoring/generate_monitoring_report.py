from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Customer_Service_Training_Dataset.csv"
)

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
# 2. Original production-model test performance
# ---------------------------------------------------------

# These values come from our untouched final test-set
# evaluation performed during Week 2.
BASELINE_TEST_ACCURACY = 0.9976
BASELINE_TEST_MACRO_F1 = 0.9975


# ---------------------------------------------------------
# 3. Reusable recommendation function
# ---------------------------------------------------------

def determine_recommendation(
    input_drift_status: str,
    prediction_drift_status: str,
    performance_status: str,
) -> str:
    """
    Decide the operational action based on monitoring signals.

    RETRAIN:
        Performance degradation is HIGH and there is also
        strong evidence of input or prediction drift.

    REVIEW:
        At least one monitoring signal is concerning,
        but retraining is not yet sufficiently justified.

    NO_ACTION:
        All monitoring signals remain within expected limits.
    """

    # Recommend retraining only when model performance has
    # clearly degraded AND another drift signal supports it.
    if (
        performance_status == "HIGH"
        and (
            input_drift_status == "HIGH"
            or prediction_drift_status == "HIGH"
        )
    ):
        return "RETRAIN"

    # If any individual signal is concerning but the evidence
    # is not yet strong enough for retraining, trigger review.
    if (
        performance_status in {"MODERATE", "HIGH"}
        or input_drift_status in {"MODERATE", "HIGH"}
        or prediction_drift_status in {"MODERATE", "HIGH"}
    ):
        return "REVIEW"

    return "NO_ACTION"


# ---------------------------------------------------------
# 4. Prediction distribution helper
# ---------------------------------------------------------

def prediction_distribution(
    predictions,
    all_intents,
):
    """
    Calculate the fraction of predictions assigned
    to every trained intent.

    Reindexing ensures that intents with zero predictions
    are still included with probability 0.
    """

    return (
        pd.Series(predictions)
        .value_counts(normalize=True)
        .reindex(
            all_intents,
            fill_value=0.0,
        )
    )


# ---------------------------------------------------------
# 5. Main monitoring workflow
# ---------------------------------------------------------

def generate_monitoring_report():
    """
    Generate the complete monitoring report using:

    1. Input-language drift
    2. Prediction-distribution drift
    3. Ground-truth performance degradation
    4. Final operational recommendation
    """

    # -----------------------------------------------------
    # Load datasets and production model
    # -----------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    drift_df = pd.read_csv(
        DRIFT_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    # Extract the fitted TF-IDF vectorizer and classifier
    # from our saved sklearn Pipeline.
    tfidf = model.named_steps[
        "tfidf"
    ]

    classifier = model.named_steps[
        "classifier"
    ]


    # -----------------------------------------------------
    # 6. INPUT LANGUAGE DRIFT
    # -----------------------------------------------------

    # Transform all text using the exact TF-IDF vocabulary
    # and IDF statistics learned during model training.
    train_vectors = tfidf.transform(
        train_df["utterance"]
    )

    validation_vectors = tfidf.transform(
        validation_df["utterance"]
    )

    drift_vectors = tfidf.transform(
        drift_df["utterance"]
    )


    # For every validation utterance, calculate similarity
    # against all training utterances and keep the highest
    # similarity score.
    #
    # This represents how close normal validation traffic is
    # to the known training-language distribution.
    validation_similarity = cosine_similarity(
        validation_vectors,
        train_vectors,
    ).max(axis=1)


    # Use the 5th percentile of normal validation similarity
    # as our lower reference boundary.
    #
    # Around 95% of normal validation utterances should be
    # above this threshold.
    similarity_threshold = np.percentile(
        validation_similarity,
        5,
    )


    # Calculate the same nearest-training similarity for
    # the new/drift dataset.
    drift_similarity = cosine_similarity(
        drift_vectors,
        train_vectors,
    ).max(axis=1)


    # Calculate the percentage of new utterances that fall
    # outside the normal reference boundary.
    drift_low_similarity_rate = np.mean(
        drift_similarity
        < similarity_threshold
    )


    # Operational drift bands used for this project.
    #
    # These are demonstration thresholds, not universal
    # industry standards.
    if drift_low_similarity_rate <= 0.10:
        input_drift_status = "LOW"

    elif drift_low_similarity_rate <= 0.25:
        input_drift_status = "MODERATE"

    else:
        input_drift_status = "HIGH"


    # -----------------------------------------------------
    # 7. PREDICTION DISTRIBUTION DRIFT
    # -----------------------------------------------------

    # Generate predictions for normal validation traffic.
    validation_predictions = model.predict(
        validation_df["utterance"]
    )

    # Generate predictions for new/drift traffic.
    drift_predictions = model.predict(
        drift_df["utterance"]
    )


    # Retrieve the complete set of trained intent classes.
    all_intents = classifier.classes_


    # Build prediction distributions for reference
    # and current traffic.
    reference_distribution = prediction_distribution(
        validation_predictions,
        all_intents,
    )

    current_distribution = prediction_distribution(
        drift_predictions,
        all_intents,
    )


    # Total Variation Distance measures how different
    # two categorical probability distributions are.
    #
    # TVD = 0 -> identical distributions
    # TVD = 1 -> completely different distributions
    prediction_tvd = 0.5 * np.abs(
        reference_distribution.values
        - current_distribution.values
    ).sum()


    # Operational prediction-drift thresholds.
    if prediction_tvd <= 0.10:
        prediction_drift_status = "LOW"

    elif prediction_tvd <= 0.20:
        prediction_drift_status = "MODERATE"

    else:
        prediction_drift_status = "HIGH"


    # -----------------------------------------------------
    # 8. PERFORMANCE MONITORING
    # -----------------------------------------------------

    # Unlike input and prediction drift, this section requires
    # ground-truth labels.
    #
    # In our controlled drift experiment, the labels are known.
    y_true = drift_df["intent"]


    # Measure model performance on the drift dataset.
    drift_accuracy = accuracy_score(
        y_true,
        drift_predictions,
    )

    drift_macro_f1 = f1_score(
        y_true,
        drift_predictions,
        average="macro",
    )


    # Compare current performance against the untouched
    # final test-set baseline.
    accuracy_drop = (
        BASELINE_TEST_ACCURACY
        - drift_accuracy
    )

    macro_f1_drop = (
        BASELINE_TEST_MACRO_F1
        - drift_macro_f1
    )


    # Operational performance degradation thresholds.
    #
    # < 0.05     -> LOW
    # 0.05-0.10  -> MODERATE
    # >= 0.10    -> HIGH
    if macro_f1_drop < 0.05:
        performance_status = "LOW"

    elif macro_f1_drop < 0.10:
        performance_status = "MODERATE"

    else:
        performance_status = "HIGH"


    # -----------------------------------------------------
    # 9. Final monitoring recommendation
    # -----------------------------------------------------

    recommendation = determine_recommendation(
        input_drift_status=
            input_drift_status,
        prediction_drift_status=
            prediction_drift_status,
        performance_status=
            performance_status,
    )


    # -----------------------------------------------------
    # 10. Print monitoring report
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CUSTOMER INTENT MODEL - MONITORING REPORT"
    )

    print(
        "=" * 72
    )


    print(
        "\nINPUT LANGUAGE DRIFT"
    )

    print(
        "-" * 72
    )

    print(
        f"Reference similarity threshold : "
        f"{similarity_threshold:.4f}"
    )

    print(
        f"New samples below threshold    : "
        f"{drift_low_similarity_rate:.2%}"
    )

    print(
        f"Input drift status             : "
        f"{input_drift_status}"
    )


    print(
        "\nPREDICTION DISTRIBUTION DRIFT"
    )

    print(
        "-" * 72
    )

    print(
        f"Total Variation Distance       : "
        f"{prediction_tvd:.4f}"
    )

    print(
        f"Prediction drift status        : "
        f"{prediction_drift_status}"
    )


    print(
        "\nMODEL PERFORMANCE"
    )

    print(
        "-" * 72
    )

    print(
        f"Baseline Test Accuracy         : "
        f"{BASELINE_TEST_ACCURACY:.4f}"
    )

    print(
        f"Current Accuracy               : "
        f"{drift_accuracy:.4f}"
    )

    print(
        f"Accuracy Drop                  : "
        f"{accuracy_drop:.4f}"
    )

    print()

    print(
        f"Baseline Test Macro F1         : "
        f"{BASELINE_TEST_MACRO_F1:.4f}"
    )

    print(
        f"Current Macro F1               : "
        f"{drift_macro_f1:.4f}"
    )

    print(
        f"Macro F1 Drop                  : "
        f"{macro_f1_drop:.4f}"
    )

    print(
        f"Performance status             : "
        f"{performance_status}"
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL MONITORING DECISION"
    )

    print(
        "=" * 72
    )

    print(
        f"Input Drift       : "
        f"{input_drift_status}"
    )

    print(
        f"Prediction Drift  : "
        f"{prediction_drift_status}"
    )

    print(
        f"Performance Drift : "
        f"{performance_status}"
    )

    print()

    print(
        f"Recommended Action: "
        f"{recommendation}"
    )

    print(
        "=" * 72
    )


    # -----------------------------------------------------
    # 11. Return results for possible future reuse
    # -----------------------------------------------------

    # Returning the metrics makes this function reusable
    # later from another monitoring service or dashboard.
    return {
        "similarity_threshold":
            similarity_threshold,

        "drift_low_similarity_rate":
            drift_low_similarity_rate,

        "input_drift_status":
            input_drift_status,

        "prediction_tvd":
            prediction_tvd,

        "prediction_drift_status":
            prediction_drift_status,

        "drift_accuracy":
            drift_accuracy,

        "accuracy_drop":
            accuracy_drop,

        "drift_macro_f1":
            drift_macro_f1,

        "macro_f1_drop":
            macro_f1_drop,

        "performance_status":
            performance_status,

        "recommendation":
            recommendation,
    }


# ---------------------------------------------------------
# 12. Run report only when executed directly
# ---------------------------------------------------------

# This is important for testing.
#
# When pytest imports determine_recommendation(), the whole
# drift-monitoring calculation will NOT automatically run.
#
# It runs only when we execute:
#
# uv run python .../generate_monitoring_report.py

if __name__ == "__main__":
    generate_monitoring_report()