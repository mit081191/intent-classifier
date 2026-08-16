from pathlib import Path

import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------
# 1. Define project and dataset paths
# ---------------------------------------------------------

# Resolve the project root dynamically so the script works
# regardless of where the repository is cloned.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = RAW_DATA_DIR / "Customer_Service_Training_Dataset.csv"
VALIDATION_FILE = RAW_DATA_DIR / "Customer_Service_Validation_Dataset.csv"

# Directory where Hugging Face Trainer can store temporary
# checkpoints and training outputs.
TRANSFORMER_OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "distilbert_training"
)


# ---------------------------------------------------------
# 2. Define transformer configuration
# ---------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 32
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 2


# ---------------------------------------------------------
# 3. Configure MLflow experiment
# ---------------------------------------------------------

# Use the same MLflow experiment as our classical models.
# This lets us compare Logistic Regression, Linear SVM,
# and DistilBERT in one place.
mlflow.set_experiment(
    "customer-intent-classification"
)


# ---------------------------------------------------------
# 4. Load training and validation datasets
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print("Training dataset:", train_df.shape)
print("Validation dataset:", validation_df.shape)


# ---------------------------------------------------------
# 5. Create intent-label mappings
# ---------------------------------------------------------

# Transformer classification models expect numeric labels.
#
# Our dataset currently contains strings such as:
#
# cancel_order
# track_refund
# payment_issue
#
# We therefore map the 27 intent strings to integer IDs.

intent_labels = sorted(
    train_df["intent"].unique()
)

label2id = {
    label: index
    for index, label in enumerate(intent_labels)
}

id2label = {
    index: label
    for label, index in label2id.items()
}

print("\nNumber of intent classes:", len(intent_labels))

print("\nSample label mappings:")
for label in intent_labels[:5]:
    print(
        f"{label} -> {label2id[label]}"
    )


# ---------------------------------------------------------
# 6. Add numeric labels to the datasets
# ---------------------------------------------------------

train_df["label"] = (
    train_df["intent"]
    .map(label2id)
)

validation_df["label"] = (
    validation_df["intent"]
    .map(label2id)
)


# ---------------------------------------------------------
# 7. Convert pandas dataframes to Hugging Face datasets
# ---------------------------------------------------------

# We only need:
#
# utterance -> model input text
# label     -> numeric target
#
# category and tags are not used as model inputs because
# our task is to predict intent from customer text alone.

train_dataset = Dataset.from_pandas(
    train_df[
        ["utterance", "label"]
    ],
    preserve_index=False,
)

validation_dataset = Dataset.from_pandas(
    validation_df[
        ["utterance", "label"]
    ],
    preserve_index=False,
)


# ---------------------------------------------------------
# 8. Load the pretrained DistilBERT tokenizer
# ---------------------------------------------------------

# The tokenizer converts natural text into the token IDs
# expected by DistilBERT.
#
# Unlike our TF-IDF pipeline, we do NOT apply clean_text()
# here. We want the pretrained tokenizer to see the original
# language as naturally as possible.

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ---------------------------------------------------------
# 9. Define tokenization function
# ---------------------------------------------------------

def tokenize_batch(batch):
    """
    Convert batches of customer utterances into DistilBERT
    token IDs and attention masks.
    """

    return tokenizer(
        batch["utterance"],

        # Pad shorter inputs so examples inside a batch
        # have a consistent sequence length.
        padding="max_length",

        # Truncate anything longer than MAX_LENGTH.
        truncation=True,

        # Our EDA showed that utterances are very short,
        # so 32 tokens should provide ample room.
        max_length=MAX_LENGTH,
    )


# ---------------------------------------------------------
# 10. Tokenize training and validation datasets
# ---------------------------------------------------------

tokenized_train = train_dataset.map(
    tokenize_batch,
    batched=True,
)

tokenized_validation = validation_dataset.map(
    tokenize_batch,
    batched=True,
)


# Remove the raw text column before passing data to Trainer.
tokenized_train = tokenized_train.remove_columns(
    ["utterance"]
)

tokenized_validation = (
    tokenized_validation.remove_columns(
        ["utterance"]
    )
)


# ---------------------------------------------------------
# 11. Load pretrained DistilBERT for classification
# ---------------------------------------------------------

# The pretrained language model is loaded with a new
# classification head containing 27 output classes.
#
# During fine-tuning, both the pretrained transformer
# representation and classification head can adapt to
# our customer-intent dataset.

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(intent_labels),
    label2id=label2id,
    id2label=id2label,
)


# ---------------------------------------------------------
# 12. Define validation metrics
# ---------------------------------------------------------

def compute_metrics(eval_prediction):
    """
    Calculate the same validation metrics used for our
    classical ML models so comparisons remain consistent.
    """

    logits, labels = eval_prediction

    # The class with the highest output score becomes
    # the predicted intent.
    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------
# 13. Define Hugging Face training configuration
# ---------------------------------------------------------

# Trainer provides the complete PyTorch training/evaluation
# loop for Hugging Face models. Hugging Face documents this
# TrainingArguments + Trainer pattern as the standard
# fine-tuning workflow.
#
# We deliberately start with a small CPU-friendly experiment.

training_args = TrainingArguments(
    output_dir=str(
        TRANSFORMER_OUTPUT_DIR
    ),

    # Evaluate the model after each epoch.
    eval_strategy="epoch",

    # Also save a checkpoint after each epoch.
    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=NUM_EPOCHS,

    # Mild weight decay is commonly used when fine-tuning
    # transformer models.
    weight_decay=0.01,

    # Keep console logs reasonably frequent.
    logging_steps=50,

    # We do not need to upload anything to Hugging Face Hub.
    push_to_hub=False,

    # Avoid automatically reporting to third-party systems.
    # We handle MLflow logging ourselves below.
    report_to="none",
)

# ---------------------------------------------------------
# 14. Create the Hugging Face Trainer
# ---------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    compute_metrics=compute_metrics,
)


# ---------------------------------------------------------
# 15. Start MLflow transformer experiment
# ---------------------------------------------------------

with mlflow.start_run(
    run_name="distilbert-finetuned"
):

    # -----------------------------------------------------
    # Log transformer hyperparameters
    # -----------------------------------------------------

    mlflow.log_param(
        "model_type",
        "DistilBERT",
    )

    mlflow.log_param(
        "pretrained_model",
        MODEL_NAME,
    )

    mlflow.log_param(
        "learning_rate",
        LEARNING_RATE,
    )

    mlflow.log_param(
        "batch_size",
        BATCH_SIZE,
    )

    mlflow.log_param(
        "epochs",
        NUM_EPOCHS,
    )

    mlflow.log_param(
        "max_length",
        MAX_LENGTH,
    )

    mlflow.log_param(
        "number_of_intents",
        len(intent_labels),
    )

    mlflow.log_param(
        "training_rows",
        len(train_df),
    )

    mlflow.log_param(
        "validation_rows",
        len(validation_df),
    )


    # -----------------------------------------------------
    # 16. Fine-tune DistilBERT
    # -----------------------------------------------------

    print(
        "\nStarting DistilBERT fine-tuning..."
    )

    trainer.train()

    print(
        "\nDistilBERT training completed."
    )


    # -----------------------------------------------------
    # 17. Evaluate on validation dataset
    # -----------------------------------------------------

    validation_results = trainer.evaluate()

    validation_accuracy = (
        validation_results["eval_accuracy"]
    )

    validation_macro_f1 = (
        validation_results["eval_macro_f1"]
    )


    print("\n" + "=" * 50)
    print("DISTILBERT VALIDATION RESULTS")
    print("=" * 50)

    print(
        f"Accuracy : "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{validation_macro_f1:.4f}"
    )


    # -----------------------------------------------------
    # 18. Log validation metrics to MLflow
    # -----------------------------------------------------

    mlflow.log_metric(
        "validation_accuracy",
        validation_accuracy,
    )

    mlflow.log_metric(
        "validation_macro_f1",
        validation_macro_f1,
    )


    # -----------------------------------------------------
    # 19. Save the fine-tuned model locally
    # -----------------------------------------------------

    final_model_dir = (
        PROJECT_ROOT
        / "models"
        / "distilbert_final"
    )

    trainer.save_model(
        str(final_model_dir)
    )

    tokenizer.save_pretrained(
        str(final_model_dir)
    )

    print(
        f"\nFine-tuned model saved to: "
        f"{final_model_dir}"
    )

# ---------------------------------------------------------
# Log saved DistilBERT files as MLflow artifacts
# ---------------------------------------------------------
#
# We log the already-saved Hugging Face model directory
# as regular artifacts instead of using
# mlflow.transformers.log_model().
#
# This avoids MLflow trying to infer optional vision
# dependencies such as torchvision, which our NLP model
# does not require.

mlflow.log_artifacts(
    local_dir=str(final_model_dir),
    artifact_path="distilbert_model",
)