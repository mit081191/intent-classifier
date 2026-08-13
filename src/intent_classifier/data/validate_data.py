from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = RAW_DATA_DIR / "Customer_Service_Training_Dataset.csv"
VALIDATION_FILE = RAW_DATA_DIR / "Customer_Service_Validation_Dataset.csv"
TEST_FILE = RAW_DATA_DIR / "Customer_Service_Testing_Dataset.csv"

EXPECTED_COLUMNS = ["utterance", "intent", "category", "tags"]


train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)


def validate_dataset(name: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 50}")
    print(f"{name} DATASET")
    print(f"{'=' * 50}")

    # Dataset size
    print(f"Shape: {df.shape}")

    # Schema validation
    print("\nColumns:")
    print(df.columns.tolist())

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        print(f"Missing expected columns: {missing_columns}")
    else:
        print("All expected columns are present.")

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Empty utterances
    empty_utterances = df["utterance"].fillna("").str.strip().eq("").sum()
    print(f"\nEmpty utterances: {empty_utterances}")

    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_rows}")

    # Unique intents
    print(f"\nUnique intents: {df['intent'].nunique()}")

    # Unique categories
    print(f"Unique categories: {df['category'].nunique()}")

    # Intent distribution
    print("\nIntent distribution:")
    print(df["intent"].value_counts().sort_index())


validate_dataset("TRAINING", train_df)
validate_dataset("VALIDATION", validation_df)
validate_dataset("TESTING", test_df)

train_intents = set(train_df["intent"].unique())
validation_intents = set(validation_df["intent"].unique())
test_intents = set(test_df["intent"].unique())

print("\n" + "=" * 50)
print("INTENT CONSISTENCY CHECK")
print("=" * 50)

print("Intents in validation but not in training:")
print(validation_intents - train_intents)

print("\nIntents in testing but not in training:")
print(test_intents - train_intents)

print("\nIntents in training but not in validation:")
print(train_intents - validation_intents)

print("\nIntents in training but not in testing:")
print(train_intents - test_intents)

if train_intents == validation_intents == test_intents:
    print("\nAll three datasets contain the same intent labels.")
else:
    print("\nIntent labels are inconsistent across datasets.")