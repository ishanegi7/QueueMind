# QueueMind Deployment Guide

This document outlines the current local deployment setup and the planned Google Cloud architecture.

## Current State (Local & Docker-Ready)

The QueueMind application is currently structured for local development and is **ready for containerization**, but is **not yet deployed** to a production cloud environment.

### 1. Local Backend Setup
The FastAPI backend can be run directly via `uvicorn`:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Local Frontend Setup
The Next.js frontend is best run directly on the host for hot-reloading:
```bash
cd frontend
npm install

# Ensure you have a .env.local with:
# NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

npm run dev
```

### 3. Environment Variables
Backend configuration is managed via `.env`:
* `API_HOST`: Host interface (default `0.0.0.0`).
* `API_PORT` / `PORT`: Port for the API. (Cloud Run provides `PORT`).
* `ENVIRONMENT`: Runtime environment (`development`, `production`).
* `LOG_LEVEL`: Logging verbosity.
* `ALLOWED_ORIGINS`: CORS origins for the frontend.
* `PATIENT_FLOW_MODEL_PATH` & `CONGESTION_MODEL_PATH`: Paths to trained `.joblib` models.

### 4. Docker Build & Run (Local Integration)
You can containerize the backend using the provided `Dockerfile`:
```bash
docker build -t queuemind-api:local .
docker run -p 8000:8000 -e PORT=8000 queuemind-api:local
```
Alternatively, use Docker Compose:
```bash
docker-compose up backend
```

## Model Artifact Requirements & Strategy
Trained machine learning models (e.g., XGBoost `.joblib` files) are required for full prediction capability. 
* **Current Strategy**: Models are loaded from the local filesystem (e.g., `models/`).
* **Git Policy**: We do **not** commit trained models to Git, as they are large binary artifacts. 
* **Future Strategy (Google Cloud)**: Model artifacts will be stored in a Google Cloud Storage (GCS) bucket. When the Cloud Run container starts, an entrypoint script or init process will download the artifacts to the container's ephemeral filesystem before launching `uvicorn`.

## Security Notes
* No GCP service account keys, API keys, or production database URLs are hardcoded in the codebase.
* `.env` files and `node_modules` are properly ignored in `.gitignore` and `.dockerignore`.
* The `/health` endpoint exposes basic readiness state but does not leak internal data.

## Planned Google Cloud Architecture (FUTURE)
* **Compute**: Google Cloud Run (Serverless containers for the FastAPI backend).
* **Storage**: Google Cloud Storage (for model artifact hosting).
* **Data Integration**: BigQuery (for historical ED analytics).
* **LLM Integration**: Gemini / Vertex AI.
* **Frontend Hosting**: Vercel or Firebase Hosting.

**Note**: None of these cloud services (Cloud Run, GCS, BigQuery, Firebase, Gemini) are currently integrated. This repository provides only the foundation.
