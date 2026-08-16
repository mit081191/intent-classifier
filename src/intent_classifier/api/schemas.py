from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """
    Request body accepted by the prediction API.
    """

    # Limit the input length as a basic API safeguard.
    # Our training utterances are very short, but we allow
    # significantly longer real-world messages.
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Customer support utterance to classify.",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """
        Reject inputs containing only whitespace.
        """

        # "   " technically has length > 0, so min_length
        # alone would not reject it.
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "Text must contain at least one non-whitespace character."
            )

        return cleaned_value


class PredictionResponse(BaseModel):
    """
    Response returned by the prediction API.
    """

    intent: str