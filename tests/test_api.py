from fastapi.testclient import TestClient

from intent_classifier.api.main import app


# ---------------------------------------------------------
# Create FastAPI test client
# ---------------------------------------------------------

# TestClient lets us call the FastAPI endpoints directly
# without manually starting Uvicorn.
client = TestClient(app)


# ---------------------------------------------------------
# 1. Test health endpoint
# ---------------------------------------------------------

def test_health_check():
    """
    Verify that the application health endpoint is available.
    """

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


# ---------------------------------------------------------
# 2. Test successful prediction
# ---------------------------------------------------------

def test_predict_intent():
    """
    Verify that a valid customer utterance produces
    the expected intent.
    """

    response = client.post(
        "/predict",
        json={
            "text": "where is my order?"
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "intent": "track_order"
    }


# ---------------------------------------------------------
# 3. Test empty input validation
# ---------------------------------------------------------

def test_empty_text():
    """
    Empty customer text should be rejected by Pydantic.
    """

    response = client.post(
        "/predict",
        json={
            "text": ""
        },
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# 4. Test whitespace-only input
# ---------------------------------------------------------

def test_whitespace_only_text():
    """
    Whitespace-only text should be rejected by our
    custom Pydantic validator.
    """

    response = client.post(
        "/predict",
        json={
            "text": "     "
        },
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# 5. Test missing text field
# ---------------------------------------------------------

def test_missing_text():
    """
    Requests without the required text field should
    be rejected automatically.
    """

    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422