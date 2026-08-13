from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from intent_classifier.features.text_preprocessing import clean_text


# Find the root of the project dynamically.
# This avoids hardcoding local paths like D:\BITS_Certification\...
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = RAW_DATA_DIR / "Customer_Service_Training_Dataset.csv"
VALIDATION_FILE = RAW_DATA_DIR / "Customer_Service_Validation_Dataset.csv"
TEST_FILE = RAW_DATA_DIR / "Customer_Service_Testing_Dataset.csv"


# Load the three predefined dataset splits.
train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)


# Apply our lightweight text-cleaning function.
# We create a new column instead of overwriting the original utterance,
# so the raw text remains available for comparison/debugging.
train_df["cleaned_utterance"] = train_df["utterance"].apply(clean_text)
validation_df["cleaned_utterance"] = validation_df["utterance"].apply(clean_text)
test_df["cleaned_utterance"] = test_df["utterance"].apply(clean_text)


# Create the TF-IDF vectorizer.
#
# max_features limits the number of vocabulary features generated.
# We are starting with 5000 as a reasonable baseline for this small dataset.
#
# ngram_range=(1, 2) means:
#   1-word patterns -> "refund"
#   2-word patterns -> "track refund"
#
# Bigrams can be useful for intent classification because short phrases
# often carry more meaning than individual words.
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)


# IMPORTANT:
# Fit the vectorizer ONLY on the training dataset.
#
# This prevents information from validation or testing data
# leaking into the feature-generation process.
X_train = vectorizer.fit_transform(
    train_df["cleaned_utterance"]
)


# Validation and testing data use the vocabulary learned from training.
# We transform them, but we do NOT fit the vectorizer again.
X_validation = vectorizer.transform(
    validation_df["cleaned_utterance"]
)

X_test = vectorizer.transform(
    test_df["cleaned_utterance"]
)


# Our target variable is the customer-service intent.
y_train = train_df["intent"]
y_validation = validation_df["intent"]
y_test = test_df["intent"]


# Print useful information to verify that feature generation worked.
print("Training feature shape:", X_train.shape)
print("Validation feature shape:", X_validation.shape)
print("Testing feature shape:", X_test.shape)

print("\nNumber of TF-IDF features:", len(vectorizer.get_feature_names_out()))

print("\nSample TF-IDF features:")
print(vectorizer.get_feature_names_out()[:30])