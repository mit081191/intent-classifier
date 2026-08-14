from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from intent_classifier.features.text_preprocessing import clean_text


# ---------------------------------------------------------
# 1. Define project and dataset paths
# ---------------------------------------------------------

# Resolve the project root dynamically so the script works
# even when another teammate clones the repository elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = RAW_DATA_DIR / "Customer_Service_Training_Dataset.csv"
VALIDATION_FILE = RAW_DATA_DIR / "Customer_Service_Validation_Dataset.csv"


# ---------------------------------------------------------
# 2. Configure MLflow experiment
# ---------------------------------------------------------

# Use the same experiment as Logistic Regression so that
# both classical ML models can be compared side-by-side.
mlflow.set_experiment("customer-intent-classification")


# ---------------------------------------------------------
# 3. Load the predefined train and validation datasets
# ---------------------------------------------------------

# We continue to leave the test dataset untouched.
# It will only be used after final model selection.
train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print("Training dataset:", train_df.shape)
print("Validation dataset:", validation_df.shape)


# ---------------------------------------------------------
# 4. Separate input text and target labels
# ---------------------------------------------------------

# Customer utterances are the model inputs.
X_train = train_df["utterance"]
X_validation = validation_df["utterance"]

# Intent is the multiclass prediction target.
y_train = train_df["intent"]
y_validation = validation_df["intent"]


# ---------------------------------------------------------
# 5. Build TF-IDF + Linear SVM pipeline
# ---------------------------------------------------------

# We intentionally use the SAME TF-IDF configuration as
# Logistic Regression so the classifier is the main variable
# being changed between the two experiments.
svm_pipeline = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                # Apply our lightweight preprocessing function.
                preprocessor=clean_text,

                # Use both individual words and two-word phrases.
                ngram_range=(1, 2),

                # Maximum allowed vocabulary size.
                max_features=5000,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                # C controls the trade-off between a wider margin
                # and classification errors.
                C=1.0,

                # Makes the experiment reproducible.
                random_state=42,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 6. Start the MLflow run
# ---------------------------------------------------------

# One MLflow run represents this complete Linear SVM experiment.
with mlflow.start_run(run_name="linear-svm"):

    # -----------------------------------------------------
    # 7. Log experiment parameters
    # -----------------------------------------------------

    # Log the important model and feature-engineering settings.
    mlflow.log_param("model_type", "LinearSVC")
    mlflow.log_param("tfidf_max_features", 5000)
    mlflow.log_param("tfidf_ngram_range", "1-2")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("random_state", 42)

    # Also log basic dataset information so that future runs
    # can be compared against the same data configuration.
    mlflow.log_param("training_rows", len(train_df))
    mlflow.log_param("validation_rows", len(validation_df))
    mlflow.log_param(
        "number_of_intents",
        train_df["intent"].nunique(),
    )


    # -----------------------------------------------------
    # 8. Train the Linear SVM model
    # -----------------------------------------------------

    print("\nTraining Linear SVM model...")

    svm_pipeline.fit(
        X_train,
        y_train,
    )

    print("Training completed.")


    # -----------------------------------------------------
    # 9. Predict validation intents
    # -----------------------------------------------------

    # The pipeline automatically performs:
    #
    # raw text
    #    ↓
    # clean_text()
    #    ↓
    # TF-IDF
    #    ↓
    # Linear SVM
    #    ↓
    # predicted intent
    y_pred = svm_pipeline.predict(
        X_validation
    )


    # -----------------------------------------------------
    # 10. Calculate validation metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_validation,
        y_pred,
    )

    # Macro F1 gives every intent equal importance,
    # regardless of class size.
    macro_f1 = f1_score(
        y_validation,
        y_pred,
        average="macro",
    )


    # -----------------------------------------------------
    # 11. Log validation metrics to MLflow
    # -----------------------------------------------------

    mlflow.log_metric(
        "validation_accuracy",
        accuracy,
    )

    mlflow.log_metric(
        "validation_macro_f1",
        macro_f1,
    )


    print("\n" + "=" * 50)
    print("LINEAR SVM VALIDATION RESULTS")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")


    # -----------------------------------------------------
    # 12. Detailed classification report
    # -----------------------------------------------------

    # Shows precision, recall and F1 for each individual intent.
    print("\nClassification Report:\n")

    report = classification_report(
        y_validation,
        y_pred,
        digits=4,
    )

    print(report)


    # -----------------------------------------------------
    # 13. Log the complete SVM pipeline to MLflow
    # -----------------------------------------------------

    # The sklearn pipeline contains our custom clean_text()
    # function. MLflow treats custom functions as untrusted
    # by default when using the safer skops model format.
    #
    # Because clean_text() is our own project code, we
    # explicitly mark it as trusted.
    mlflow.sklearn.log_model(
        sk_model=svm_pipeline,
        name="model",
        skops_trusted_types=[
            "intent_classifier.features.text_preprocessing.clean_text"
        ],
    )


    # -----------------------------------------------------
    # 14. Error analysis
    # -----------------------------------------------------

    # Store original text, correct label and predicted label.
    error_analysis = pd.DataFrame(
        {
            "utterance": X_validation,
            "actual_intent": y_validation,
            "predicted_intent": y_pred,
        }
    )

    # Keep only incorrect predictions.
    misclassified = error_analysis[
        error_analysis["actual_intent"]
        != error_analysis["predicted_intent"]
    ]

    print("\n" + "=" * 50)
    print("MISCLASSIFIED VALIDATION EXAMPLES")
    print("=" * 50)

    print(
        f"Total misclassified examples: "
        f"{len(misclassified)}"
    )

    # Log the number of mistakes as another model metric.
    mlflow.log_metric(
        "validation_misclassified_count",
        len(misclassified),
    )

    # Display complete utterances instead of truncating them.
    pd.set_option(
        "display.max_colwidth",
        None,
    )

    print(
        misclassified[
            [
                "utterance",
                "actual_intent",
                "predicted_intent",
            ]
        ].to_string(index=False)
    )