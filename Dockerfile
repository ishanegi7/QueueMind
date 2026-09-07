# Use the official Python lightweight image
# Python 3.11 is used here for stability and performance.
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set the working directory
WORKDIR /app

# Install system dependencies if required by xgboost/joblib, etc.
# libgomp1 is often required for XGBoost on linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency configuration first to cache the layer
COPY requirements.txt ./

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core backend code
# NOTE: Models should ideally be loaded from Cloud Storage at runtime
# or mapped via volume, rather than baked into the container image.
COPY api/ ./api/
COPY src/ ./src/

# Provide a default port for local testing, though Cloud Run injects $PORT
ENV PORT=8000
EXPOSE ${PORT}

# Run the FastAPI application using uvicorn
# Cloud Run overrides the PORT environment variable
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
