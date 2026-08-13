from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = RAW_DATA_DIR / "Customer_Service_Training_Dataset.csv"
VALIDATION_FILE = RAW_DATA_DIR / "Customer_Service_Validation_Dataset.csv"
TEST_FILE = RAW_DATA_DIR / "Customer_Service_Testing_Dataset.csv"


train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)


print("Training dataset shape:", train_df.shape)
print("Validation dataset shape:", validation_df.shape)
print("Testing dataset shape:", test_df.shape)

print("\nTraining columns:")
print(train_df.columns.tolist())

print("\nValidation columns:")
print(validation_df.columns.tolist())

print("\nTesting columns:")
print(test_df.columns.tolist())

print("\nTraining sample:")
print(train_df.head())

print("\nValidation sample:")
print(validation_df.head())

print("\nTesting sample:")
print(test_df.head())