# Use a lightweight Python base image.
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app


# ---------------------------------------------------------
# Install uv
# ---------------------------------------------------------

# Copy uv directly from the official uv container image.
# This avoids installing uv with pip.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# ---------------------------------------------------------
# Copy dependency files first
# ---------------------------------------------------------

# Docker can cache this layer when application code changes
# but dependencies remain the same.
COPY pyproject.toml uv.lock ./


# ---------------------------------------------------------
# Install project dependencies
# ---------------------------------------------------------

# Install dependencies into the project virtual environment.
#
# --frozen ensures uv uses the existing uv.lock exactly.
# --no-dev avoids unnecessary development dependencies.
RUN uv sync --frozen --no-dev --no-install-project
# ---------------------------------------------------------
# Copy application source code
# ---------------------------------------------------------

COPY src ./src


# ---------------------------------------------------------
# Copy selected production model
# ---------------------------------------------------------

# The FastAPI inference module expects:
#
# /app/models/linear_svm_pipeline.joblib
COPY models/linear_svm_pipeline.joblib ./models/linear_svm_pipeline.joblib


# ---------------------------------------------------------
# Make the uv environment available
# ---------------------------------------------------------

ENV PATH="/app/.venv/bin:$PATH"

# Add src to Python module search path.
ENV PYTHONPATH="/app/src"


# ---------------------------------------------------------
# Expose FastAPI port
# ---------------------------------------------------------

EXPOSE 8000


# ---------------------------------------------------------
# Start FastAPI
# ---------------------------------------------------------

CMD ["uvicorn", "intent_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]