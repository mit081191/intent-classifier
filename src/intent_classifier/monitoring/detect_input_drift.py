from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DRIFT_DATA_DIR = PROJECT_ROOT / "data" / "drift"

TRAIN_FILE = (
    RAW_DATA_DIR
    / "Customer_Service_Training_Dataset.csv"
)

VALIDATION_FILE = (
    RAW_DATA_DIR
    / "Customer_Service_Validation_Dataset.csv"
)

DRIFT_FILE = (
    DRIFT_DATA_DIR
    / "Customer_Service_Drift_Dataset.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "linear_svm_pipeline.joblib"
)


# ---------------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
drift_df = pd.read_csv(DRIFT_FILE)

print("Training rows   :", len(train_df))
print("Validation rows :", len(validation_df))
print("Drift rows      :", len(drift_df))


# ---------------------------------------------------------
# 3. Load the selected production pipeline
# ---------------------------------------------------------

# The saved pipeline contains:
#
# clean_text()
#      ↓
# fitted TF-IDF vectorizer
#      ↓
# trained Linear SVM

model_pipeline = joblib.load(
    MODEL_FILE
)


# ---------------------------------------------------------
# 4. Extract the fitted TF-IDF vectorizer
# ---------------------------------------------------------

# We want to analyse incoming language using EXACTLY the
# same vocabulary and IDF statistics used by the deployed
# production model.
tfidf_vectorizer = model_pipeline.named_steps[
    "tfidf"
]


# ---------------------------------------------------------
# 5. Transform text into the production TF-IDF space
# ---------------------------------------------------------

train_vectors = tfidf_vectorizer.transform(
    train_df["utterance"]
)

validation_vectors = tfidf_vectorizer.transform(
    validation_df["utterance"]
)

drift_vectors = tfidf_vectorizer.transform(
    drift_df["utterance"]
)


print("\nTF-IDF vector shapes:")

print(
    "Training   :",
    train_vectors.shape,
)

print(
    "Validation :",
    validation_vectors.shape,
)

print(
    "Drift      :",
    drift_vectors.shape,
)


# ---------------------------------------------------------
# 6. Find similarity to known training language
# ---------------------------------------------------------

# For each validation utterance:
#
# compare it against every training utterance
# and keep only the highest cosine similarity.
#
# This creates our REFERENCE similarity distribution.
validation_similarity_matrix = cosine_similarity(
    validation_vectors,
    train_vectors,
)

validation_max_similarity = (
    validation_similarity_matrix.max(
        axis=1
    )
)


# Repeat the same process for the new drift traffic.
drift_similarity_matrix = cosine_similarity(
    drift_vectors,
    train_vectors,
)

drift_max_similarity = (
    drift_similarity_matrix.max(
        axis=1
    )
)


# ---------------------------------------------------------
# 7. Build a reference threshold from validation data
# ---------------------------------------------------------

# Rather than inventing a similarity threshold such as 0.50,
# we derive it from the normal validation distribution.
#
# The 5th percentile means:
#
# approximately 95% of normal validation utterances have
# similarity ABOVE this value.
#
# Therefore, production utterances falling below this value
# look unusually different from our original data.

similarity_threshold = np.percentile(
    validation_max_similarity,
    5,
)


# ---------------------------------------------------------
# 8. Calculate low-similarity rates
# ---------------------------------------------------------

# Under normal/reference conditions, we expect roughly
# 5% of validation examples to fall below the threshold.

validation_low_similarity_rate = np.mean(
    validation_max_similarity
    < similarity_threshold
)


# Measure how much of the drift dataset now falls below
# that SAME reference threshold.
drift_low_similarity_rate = np.mean(
    drift_max_similarity
    < similarity_threshold
)


# ---------------------------------------------------------
# 9. Calculate summary statistics
# ---------------------------------------------------------

validation_mean_similarity = np.mean(
    validation_max_similarity
)

drift_mean_similarity = np.mean(
    drift_max_similarity
)

mean_similarity_drop = (
    validation_mean_similarity
    - drift_mean_similarity
)


# ---------------------------------------------------------
# 10. Determine an input-drift status
# ---------------------------------------------------------

# We use the percentage of recent utterances that fall
# outside our reference language boundary.
#
# These are operational monitoring thresholds:
#
# <= 10%  -> LOW
# <= 25%  -> MODERATE
# > 25%   -> HIGH
#
# The individual-text similarity boundary itself is derived
# from real validation data rather than selected arbitrarily.

if drift_low_similarity_rate <= 0.10:
    drift_status = "LOW"

elif drift_low_similarity_rate <= 0.25:
    drift_status = "MODERATE"

else:
    drift_status = "HIGH"


# ---------------------------------------------------------
# 11. Print reference statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("REFERENCE LANGUAGE SIMILARITY")
print("=" * 70)

print(
    f"Validation Mean Max Similarity : "
    f"{validation_mean_similarity:.4f}"
)

print(
    f"5th Percentile Threshold       : "
    f"{similarity_threshold:.4f}"
)

print(
    f"Validation Below Threshold     : "
    f"{validation_low_similarity_rate:.2%}"
)


# ---------------------------------------------------------
# 12. Print drift statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NEW / DRIFT LANGUAGE SIMILARITY")
print("=" * 70)

print(
    f"Drift Mean Max Similarity : "
    f"{drift_mean_similarity:.4f}"
)

print(
    f"Mean Similarity Drop      : "
    f"{mean_similarity_drop:.4f}"
)

print(
    f"Drift Below Threshold     : "
    f"{drift_low_similarity_rate:.2%}"
)


# ---------------------------------------------------------
# 13. Overall drift decision
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INPUT DRIFT ASSESSMENT")
print("=" * 70)

print(
    f"Input Drift Status : "
    f"{drift_status}"
)


# ---------------------------------------------------------
# 14. Inspect the most unfamiliar utterances
# ---------------------------------------------------------

# Add the similarity score to a copy of the drift data.
drift_analysis = drift_df.copy()

drift_analysis[
    "max_training_similarity"
] = drift_max_similarity


# Mark utterances that fall outside our normal/reference
# language boundary.
drift_analysis[
    "below_reference_threshold"
] = (
    drift_analysis[
        "max_training_similarity"
    ]
    < similarity_threshold
)


# Sort from least similar to most similar.
most_unfamiliar = (
    drift_analysis
    .sort_values(
        "max_training_similarity",
        ascending=True,
    )
    .head(20)
)


print("\n" + "=" * 70)
print("MOST UNFAMILIAR DRIFT UTTERANCES")
print("=" * 70)

pd.set_option(
    "display.max_colwidth",
    None,
)

print(
    most_unfamiliar[
        [
            "utterance",
            "max_training_similarity",
            "below_reference_threshold",
        ]
    ].to_string(
        index=False
    )
)