from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from intent_classifier.features.text_preprocessing import clean_text


# ---------------------------------------------------------
# 1. Define project and dataset paths
# ---------------------------------------------------------

# Find the project root dynamically so the code works
# regardless of where a teammate clones the repository.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = RAW_DATA_DIR / "Customer_Service_Training_Dataset.csv"
VALIDATION_FILE = RAW_DATA_DIR / "Customer_Service_Validation_Dataset.csv"


# ---------------------------------------------------------
# 2. Configure MLflow experiment
# ---------------------------------------------------------

# All model-training runs for this project will be grouped
# under one MLflow experiment.
#
# If this experiment does not exist yet, MLflow creates it.
mlflow.set_experiment("customer-intent-classification")


# ---------------------------------------------------------
# 3. Load training and validation datasets
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print("Training dataset:", train_df.shape)
print("Validation dataset:", validation_df.shape)


# ---------------------------------------------------------
# 4. Separate input text (X) and target intent (y)
# ---------------------------------------------------------

# X contains the customer utterances that the model will read.
X_train = train_df["utterance"]
X_validation = validation_df["utterance"]

# y contains the correct intent labels that the model
# should learn to predict.
y_train = train_df["intent"]
y_validation = validation_df["intent"]


# ---------------------------------------------------------
# 5. Build the complete ML pipeline
# ---------------------------------------------------------

# Pipeline keeps text preprocessing, TF-IDF feature generation,
# and the classifier together as one reusable model object.
#
# During training and prediction:
#
# Raw text
#    ↓
# clean_text()
#    ↓
# TF-IDF
#    ↓
# Logistic Regression
#    ↓
# Intent prediction

baseline_pipeline = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                # Apply our existing lightweight cleaning function
                # before TF-IDF tokenization.
                preprocessor=clean_text,

                # Use both individual words and two-word phrases.
                ngram_range=(1, 2),

                # Maximum vocabulary size.
                # The actual number of generated features may be lower.
                max_features=5000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                # Give the optimizer enough iterations to converge.
                max_iter=1000,

                # Makes training reproducible where randomness is involved.
                random_state=42,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 6. Start MLflow experiment run
# ---------------------------------------------------------

# One MLflow run represents one specific model-training experiment.
#
# We give the run a meaningful name so that it is easy to
# identify in the MLflow UI.
with mlflow.start_run(run_name="logistic-regression-baseline"):

    # -----------------------------------------------------
    # 7. Log experiment parameters
    # -----------------------------------------------------

    # These are the important settings used for this model.
    # MLflow stores them so we can compare future experiments.
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("tfidf_max_features", 5000)
    mlflow.log_param("tfidf_ngram_range", "1-2")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("random_state", 42)

    # Log some useful information about the dataset as well.
    mlflow.log_param("training_rows", len(train_df))
    mlflow.log_param("validation_rows", len(validation_df))
    mlflow.log_param("number_of_intents", train_df["intent"].nunique())


    # -----------------------------------------------------
    # 8. Train the baseline model
    # -----------------------------------------------------

    print("\nTraining Logistic Regression baseline...")

    baseline_pipeline.fit(
        X_train,
        y_train,
    )

    print("Training completed.")


    # -----------------------------------------------------
    # 9. Predict validation intents
    # -----------------------------------------------------

    # The validation dataset is transformed using the TF-IDF
    # vocabulary learned from the training dataset.
    y_pred = baseline_pipeline.predict(
        X_validation
    )


    # -----------------------------------------------------
    # 10. Calculate validation metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_validation,
        y_pred,
    )

    # Macro F1 calculates F1 independently for every intent
    # and gives all 27 intents equal importance.
    macro_f1 = f1_score(
        y_validation,
        y_pred,
        average="macro",
    )


    # -----------------------------------------------------
    # 11. Log validation metrics to MLflow
    # -----------------------------------------------------

    # These metrics will appear in the MLflow UI and allow
    # us to compare Logistic Regression, SVM and DistilBERT.
    mlflow.log_metric(
        "validation_accuracy",
        accuracy,
    )

    mlflow.log_metric(
        "validation_macro_f1",
        macro_f1,
    )


    print("\n" + "=" * 50)
    print("BASELINE VALIDATION RESULTS")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")


    # -----------------------------------------------------
    # 12. Detailed per-intent evaluation
    # -----------------------------------------------------

    # Classification report provides precision, recall and
    # F1-score for each of the 27 customer intents.
    print("\nClassification Report:\n")

    report = classification_report(
        y_validation,
        y_pred,
        digits=4,
    )

    print(report)

    # -----------------------------------------------------
    # 13. Log the complete sklearn pipeline to MLflow
    # -----------------------------------------------------

    # The pipeline contains our custom clean_text() function.
    # MLflow's sklearn persistence layer treats custom Python
    # functions as untrusted by default for security reasons.
    #
    # Since clean_text() is our own project code and we know
    # exactly what it does, we explicitly mark it as trusted.

    mlflow.sklearn.log_model(
        sk_model=baseline_pipeline,
        name="model",
        skops_trusted_types=[
            "intent_classifier.features.text_preprocessing.clean_text"
        ],
    )