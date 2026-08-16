from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

# Resolve the project root dynamically so the script works
# regardless of where the repository is cloned.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

VALIDATION_FILE = (
    RAW_DATA_DIR / "Customer_Service_Validation_Dataset.csv"
)

# This is the fine-tuned DistilBERT model that we saved
# after training completed.
MODEL_DIR = (
    PROJECT_ROOT / "models" / "distilbert_final"
)


# ---------------------------------------------------------
# 2. Load validation dataset
# ---------------------------------------------------------

validation_df = pd.read_csv(
    VALIDATION_FILE
)

print(
    "Validation dataset:",
    validation_df.shape,
)


# ---------------------------------------------------------
# 3. Load tokenizer and fine-tuned model
# ---------------------------------------------------------

# Load the tokenizer from our saved model directory.
# This ensures we use the exact tokenizer configuration
# associated with the trained model.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

# Load the trained 27-class DistilBERT model.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

# Put the model into evaluation mode.
#
# This disables training-specific behaviour such as dropout.
model.eval()


# ---------------------------------------------------------
# 4. Read label mappings from the model configuration
# ---------------------------------------------------------

# The label mappings were saved with the model during training.
#
# Example:
#
# 0 -> cancel_order
# 1 -> change_order
# ...
id2label = {
    int(key): value
    for key, value in model.config.id2label.items()
}

label2id = model.config.label2id


# ---------------------------------------------------------
# 5. Tokenize validation utterances
# ---------------------------------------------------------

# We use the same maximum sequence length that was used
# during DistilBERT fine-tuning.
MAX_LENGTH = 32

encoded_inputs = tokenizer(
    validation_df["utterance"].tolist(),

    # Pad shorter sequences to the same length.
    padding=True,

    # Truncate anything longer than MAX_LENGTH.
    truncation=True,

    max_length=MAX_LENGTH,

    # Return PyTorch tensors because the model is
    # implemented in PyTorch.
    return_tensors="pt",
)


# ---------------------------------------------------------
# 6. Run DistilBERT predictions
# ---------------------------------------------------------

# torch.no_grad() tells PyTorch that we are only performing
# inference and do not need gradients.
#
# This reduces memory usage and speeds up prediction.
with torch.no_grad():

    outputs = model(
        **encoded_inputs
    )

    # logits contains one score for each of the 27 intents.
    logits = outputs.logits


# ---------------------------------------------------------
# 7. Convert logits into predicted label IDs
# ---------------------------------------------------------

# For every validation example, select the class with
# the highest model score.
predicted_label_ids = torch.argmax(
    logits,
    dim=1,
).cpu().numpy()


# ---------------------------------------------------------
# 8. Convert predicted IDs back to intent names
# ---------------------------------------------------------

predicted_intents = [
    id2label[int(label_id)]
    for label_id in predicted_label_ids
]


# ---------------------------------------------------------
# 9. Create error-analysis dataframe
# ---------------------------------------------------------

error_analysis = pd.DataFrame(
    {
        "utterance": validation_df["utterance"],
        "actual_intent": validation_df["intent"],
        "predicted_intent": predicted_intents,
    }
)


# ---------------------------------------------------------
# 10. Keep only misclassified examples
# ---------------------------------------------------------

misclassified = error_analysis[
    error_analysis["actual_intent"]
    != error_analysis["predicted_intent"]
].copy()


print("\n" + "=" * 60)
print("DISTILBERT MISCLASSIFIED VALIDATION EXAMPLES")
print("=" * 60)

print(
    f"Total misclassified examples: "
    f"{len(misclassified)}"
)


# Show complete utterances instead of truncating text.
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