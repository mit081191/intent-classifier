from pathlib import Path
import joblib
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
# regardless of where the repository is cloned.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = RAW_DATA_DIR / "Customer_Service_Training_Dataset.csv"
TEST_FILE = RAW_DATA_DIR / "Customer_Service_Testing_Dataset.csv"


# ---------------------------------------------------------
# 2. Configure MLflow
# ---------------------------------------------------------

# Keep the final test run in the same experiment as all
# previous model-training experiments.
mlflow.set_experiment("customer-intent-classification")


# ---------------------------------------------------------
# 3. Load training and TEST datasets
# ---------------------------------------------------------

# The test dataset has intentionally remained untouched
# until this final evaluation stage.
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Training dataset:", train_df.shape)
print("Testing dataset :", test_df.shape)


# ---------------------------------------------------------
# 4. Separate input text and target labels
# ---------------------------------------------------------

X_train = train_df["utterance"]
y_train = train_df["intent"]

X_test = test_df["utterance"]
y_test = test_df["intent"]


# ---------------------------------------------------------
# 5. Rebuild the selected final model
# ---------------------------------------------------------

# Linear SVM was selected because it produced the strongest
# validation Macro F1 while remaining very lightweight.
#
# We use exactly the same preprocessing and TF-IDF settings
# that were used during validation.
final_pipeline = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                # Apply our reusable light text cleaning.
                preprocessor=clean_text,

                # Use both unigrams and bigrams.
                ngram_range=(1, 2),

                # Same feature cap used in all classical runs.
                max_features=5000,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                # Same SVM configuration used during model selection.
                C=1.0,
                random_state=42,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 6. Start the final MLflow test run
# ---------------------------------------------------------

with mlflow.start_run(
    run_name="linear-svm-final-test"
):

    # Log the final model configuration.
    mlflow.log_param("model_type", "LinearSVC")
    mlflow.log_param("tfidf_max_features", 5000)
    mlflow.log_param("tfidf_ngram_range", "1-2")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("random_state", 42)

    mlflow.log_param("training_rows", len(train_df))
    mlflow.log_param("test_rows", len(test_df))
    mlflow.log_param(
        "number_of_intents",
        train_df["intent"].nunique(),
    )


    # -----------------------------------------------------
    # 7. Train the selected model
    # -----------------------------------------------------

    print("\nTraining selected Linear SVM model...")

    final_pipeline.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    # ---------------------------------------------------------
    # Save the trained final pipeline for application inference
    # ---------------------------------------------------------

    # Store generated model artifacts under the root-level
    # models directory.
    MODEL_DIR = PROJECT_ROOT / "models"
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_MODEL_FILE = (
            MODEL_DIR / "linear_svm_pipeline.joblib"
    )

    # Save the COMPLETE sklearn pipeline.
    #
    # This contains:
    #
    # clean_text()
    #      ↓
    # TF-IDF vectorizer
    #      ↓
    # Linear SVM
    #
    # Therefore, our inference application can later provide
    # raw customer text directly to pipeline.predict().
    joblib.dump(
        final_pipeline,
        FINAL_MODEL_FILE,
    )

    print(
        f"\nFinal model saved to: "
        f"{FINAL_MODEL_FILE}"
    )
    # -----------------------------------------------------
    # 8. Predict on the untouched TEST dataset
    # -----------------------------------------------------

    y_pred = final_pipeline.predict(
        X_test
    )


    # -----------------------------------------------------
    # 9. Calculate final test metrics
    # -----------------------------------------------------

    test_accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    test_macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )


    print("\n" + "=" * 60)
    print("FINAL LINEAR SVM TEST RESULTS")
    print("=" * 60)

    print(
        f"Test Accuracy : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Macro F1 : "
        f"{test_macro_f1:.4f}"
    )


    # -----------------------------------------------------
    # 10. Log final test metrics to MLflow
    # -----------------------------------------------------

    # Use "test_" names so these metrics are clearly
    # distinguishable from validation metrics.
    mlflow.log_metric(
        "test_accuracy",
        test_accuracy,
    )

    mlflow.log_metric(
        "test_macro_f1",
        test_macro_f1,
    )


    # -----------------------------------------------------
    # 11. Print detailed per-intent results
    # -----------------------------------------------------

    print("\nClassification Report:\n")

    report = classification_report(
        y_test,
        y_pred,
        digits=4,
    )

    print(report)


    # -----------------------------------------------------
    # 12. Perform final test error analysis
    # -----------------------------------------------------

    error_analysis = pd.DataFrame(
        {
            "utterance": X_test,
            "actual_intent": y_test,
            "predicted_intent": y_pred,
        }
    )

    # Keep only incorrect predictions.
    misclassified = error_analysis[
        error_analysis["actual_intent"]
        != error_analysis["predicted_intent"]
    ].copy()


    print("\n" + "=" * 60)
    print("MISCLASSIFIED TEST EXAMPLES")
    print("=" * 60)

    print(
        f"Total misclassified examples: "
        f"{len(misclassified)}"
    )

    # Log the number of test errors to MLflow.
    mlflow.log_metric(
        "test_misclassified_count",
        len(misclassified),
    )

    # Show complete customer utterances.
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


    # -----------------------------------------------------
    # 13. Log the final sklearn model to MLflow
    # -----------------------------------------------------

    # The pipeline includes our own clean_text() function,
    # so we explicitly mark that type as trusted.
    mlflow.sklearn.log_model(
        sk_model=final_pipeline,
        name="model",
        skops_trusted_types=[
            "intent_classifier.features.text_preprocessing.clean_text"
        ],
    )