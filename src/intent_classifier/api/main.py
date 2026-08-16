from fastapi import FastAPI

from intent_classifier.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from intent_classifier.inference.predict import predict_intent


# ---------------------------------------------------------
# 1. Create FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Customer Intent Classifier API",
    description=(
        "REST API for predicting customer-support intents "
        "using the selected TF-IDF + Linear SVM model."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# 2. Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    """
    Simple endpoint used to verify that the API is running.
    """

    return {
        "status": "healthy"
    }


# ---------------------------------------------------------
# 3. Prediction endpoint
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    """
    Predict the intent of a customer support utterance.
    """

    # Pass the raw customer text to our reusable
    # inference function.
    predicted_intent = predict_intent(
        request.text
    )

    # Return the prediction using the response schema.
    return PredictionResponse(
        intent=predicted_intent
    )