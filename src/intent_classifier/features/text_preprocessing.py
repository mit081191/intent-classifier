import re


def clean_text(text: str) -> str:
    """
    Apply light text cleaning before feature extraction.

    The goal is to normalize obvious formatting differences without
    aggressively changing the meaning of short customer utterances.
    """

    # Convert text to lowercase so words such as "Order" and "order"
    # are treated consistently by the classical ML pipeline.
    text = text.lower()

    # Replace multiple whitespace characters (spaces, tabs, newlines)
    # with a single space.
    text = re.sub(r"\s+", " ", text)

    # Remove whitespace from the beginning and end of the utterance.
    text = text.strip()

    return text

if __name__ == "__main__":
    # Small sanity check to verify that our preprocessing behaves
    # as expected before integrating it into the full pipeline.
    sample_text = "  I DON'T   know how to cancel my order!  "

    cleaned_text = clean_text(sample_text)

    print("Original:", repr(sample_text))
    print("Cleaned :", repr(cleaned_text))