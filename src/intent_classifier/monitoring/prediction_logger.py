from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Define monitoring log location
# ---------------------------------------------------------

# Resolve project root dynamically.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MONITORING_DIR = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
)

PREDICTION_LOG_FILE = (
    MONITORING_DIR
    / "prediction_log.csv"
)


# ---------------------------------------------------------
# 2. Log a model prediction
# ---------------------------------------------------------

def log_prediction(
    text: str,
    predicted_intent: str,
) -> None:
    """
    Store one production prediction for future monitoring.

    Each record contains:
    - prediction timestamp
    - original customer utterance
    - predicted intent
    """

    # Create the monitoring directory automatically
    # if it does not already exist.
    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use UTC so timestamps remain consistent if the
    # application is deployed in different environments.
    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    new_record = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "text": text,
                "predicted_intent": predicted_intent,
            }
        ]
    )

    # If the file already exists, append the new prediction.
    #
    # If this is the first prediction, also write
    # the CSV header.
    file_exists = PREDICTION_LOG_FILE.exists()

    new_record.to_csv(
        PREDICTION_LOG_FILE,
        mode="a",
        header=not file_exists,
        index=False,
    )