from pathlib import Path

import joblib


# ---------------------------------------------------------
# 1. Define model path
# ---------------------------------------------------------

# Resolve the project root dynamically so the code works
# regardless of where the repository is cloned.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "linear_svm_pipeline.joblib"
)


# ---------------------------------------------------------
# 2. Load the trained pipeline
# ---------------------------------------------------------

# The saved joblib file contains the complete pipeline:
#
# clean_text()
#      ↓
# TF-IDF
#      ↓
# Linear SVM
#
# Loading it once gives us everything required for inference.
model = joblib.load(
    MODEL_FILE
)


# ---------------------------------------------------------
# 3. Create reusable prediction function
# ---------------------------------------------------------

def predict_intent(text: str) -> str:
    """
    Predict the customer-service intent for one utterance.
    """

    # sklearn expects an iterable of samples,
    # so even one string is passed inside a list.
    prediction = model.predict(
        [text]
    )

    # model.predict() returns an array.
    # We return the first predicted label.
    return prediction[0]


# ---------------------------------------------------------
# 4. Simple local inference test
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_utterances = [
        "where is my order?",
        "I want to cancel my account",
        "how can I get my refund?",
        "I cannot remember my password",
        "I want to speak to a human agent",
    ]

    print("\nMODEL INFERENCE TEST")
    print("=" * 60)

    for utterance in sample_utterances:

        predicted_intent = predict_intent(
            utterance
        )

        print(
            f"\nText      : {utterance}"
        )

        print(
            f"Prediction: {predicted_intent}"
        )