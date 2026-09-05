# QueueMind REST API Documentation

**QueueMind** provides a high-performance, asynchronous REST API layer built on **FastAPI** and **Pydantic v2**. 

The API serves as a **thin adapter layer** around QueueMind's core machine learning, explainability, queue health scoring, and what-if simulation engines. It strictly avoids re-implementing business logic or fabricating predictions when model artifacts are unconfigured.

---

## 1. Architectural Principles

1. **Adapter Architecture**: All numerical algorithms, validation rules, SHAP explanations, conformal calibrations, and flow calculations are executed via `src/queuemind/` core modules. The API handles schema serialization, deserialization, HTTP status mapping, and route orchestration.
2. **Leakage Safety**: The API strictly forbids future timestamps, discharge destinations, total lengths of stay, or target fields in prediction requests. Any prohibited field raises an immediate `422 Unprocessable Content` response.
3. **Fail-Fast Integrity**: If a model artifact is unconfigured or unavailable on disk, the service returns `503 Service Unavailable` with a descriptive message rather than hallucinating fake predictions.
4. **Non-Causal Semantics**: All SHAP attributions, Queue Health scores, and what-if simulations include machine-readable and human-readable disclaimers highlighting that statistical attributions and deterministic simulations do not establish clinical causation.

---

## 2. Configuration & Environment Variables

The application is configured using Pydantic Settings (`api/config.py`), reading from environment variables or a `.env` file:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `API_HOST` | `str` | `"0.0.0.0"` | Host interface for Uvicorn server |
| `API_PORT` | `int` | `8000` | Port for Uvicorn server |
| `API_RELOAD` | `bool` | `false` | Enable auto-reloading in development |
| `ALLOWED_ORIGINS` | `list[str]` | `["http://localhost:3000", ...]` | Allowed CORS origins (JSON array or comma-separated) |
| `PATIENT_FLOW_MODEL_PATH` | `str \| None` | `None` | Path to serialized `PatientFlowPredictor` joblib artifact |
| `CONGESTION_MODEL_PATH` | `str \| None` | `None` | Path to serialized `CongestionPredictor` joblib artifact |
| `QUEUE_HEALTH_CAPACITY` | `float` | `50.0` | Default ED bed capacity reference |
| `QUEUE_HEALTH_ARRIVAL_REF` | `float` | `10.0` | Default 60-minute arrival velocity reference |
| `QUEUE_HEALTH_ACUITY_REF` | `float` | `0.20` | Default high-acuity patient ratio reference |
| `LOG_LEVEL` | `str` | `"INFO"` | Standard Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 3. Interactive Documentation

When the API is running, interactive documentation is available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI JSON**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 4. Endpoints Specification

### 4.1 System Health & Readiness

#### `GET /health`
Returns the operational health of the FastAPI process and the availability status of trained model artifacts.

**Response `200 OK`**:
```json
{
  "status": "healthy",
  "service": "queuemind-api",
  "version": "0.1.0",
  "models": {
    "patient_flow": "ready",
    "congestion": "unavailable"
  }
}
```

---

### 4.2 Patient-Flow Prediction

#### `POST /predict/patient-flow`
Predicts remaining patient-flow duration in minutes for a patient at a given decision snapshot. Supports optional split-conformal prediction intervals and local TreeSHAP feature attributions.

**Strict Leakage Guard**: Prohibits `stay_id`, `intime`, `outtime`, `disposition`, `los_hours`, `remaining_time_minutes`, etc.

**Request Body (`PatientFlowRequest`)**:
```json
{
  "acuity": 3,
  "temperature": 98.6,
  "heartrate": 82.0,
  "resprate": 18.0,
  "o2sat": 98.0,
  "sbp": 124.0,
  "dbp": 78.0,
  "pain": 4.0,
  "age": 52,
  "gender": "F",
  "arrival_transport": "WALK IN",
  "current_active_census": 42.0,
  "coverage_level": 0.90,
  "return_explanation": true
}
```

**Response `200 OK` (`PatientFlowResponse`)**:
```json
{
  "predicted_remaining_time_minutes": 165.4,
  "unit": "minutes",
  "model_name": "xgboost_patient_flow",
  "model_version": "1.0.0",
  "prediction_interval": {
    "lower_minutes": 112.1,
    "upper_minutes": 228.6,
    "coverage_level": 0.90,
    "method": "split_conformal",
    "non_negative_enforced": true
  },
  "explanation": {
    "prediction": 165.4,
    "base_value": 180.2,
    "features": [
      {
        "name": "acuity",
        "value": 3,
        "attribution": 14.5,
        "direction": "increases_time",
        "rank": 1
      },
      {
        "name": "current_active_census",
        "value": 42.0,
        "attribution": 9.2,
        "direction": "increases_time",
        "rank": 2
      }
    ],
    "interpretation": "SHAP values explain model behavior for the prediction; they do not establish causality."
  },
  "non_clinical_disclaimer": "QueueMind is an operational patient-flow forecasting prototype. Predictions do not constitute medical advice or clinical triage decisions."
}
```

**Error Responses**:
- `422 Unprocessable Content`: If input features violate physiological boundaries or contain prohibited leakage fields.
- `503 Service Unavailable`: If `PATIENT_FLOW_MODEL_PATH` is not configured or the model is missing.

---

### 4.3 Department Congestion Forecasting

#### `POST /predict/congestion`
Projects future active patient census across +30m, +60m, and +120m tactical planning horizons. Computes conformal bounds, situational congestion states, and queue dynamics bottleneck signals.

**Request Body (`CongestionForecastRequest`)**:
```json
{
  "current_active_census": 42.0,
  "recent_arrivals_15m": 4.0,
  "recent_arrivals_30m": 7.0,
  "recent_arrivals_60m": 12.0,
  "recent_arrivals_120m": 22.0,
  "recent_departures_15m": 3.0,
  "recent_departures_30m": 5.0,
  "recent_departures_60m": 10.0,
  "recent_departures_120m": 20.0,
  "high_acuity_ratio": 0.25,
  "coverage_level": 0.90
}
```

**Response `200 OK` (`CongestionForecastResponse`)**:
```json
{
  "horizons": {
    "30m": {
      "horizon_minutes": 30,
      "predicted_census": 44.5,
      "congestion_state": "MODERATE",
      "prediction_interval": {
        "lower_census": 38.0,
        "upper_census": 51.0,
        "coverage_level": 0.90,
        "method": "split_conformal"
      }
    },
    "60m": {
      "horizon_minutes": 60,
      "predicted_census": 47.2,
      "congestion_state": "BUSY",
      "prediction_interval": {
        "lower_census": 39.5,
        "upper_census": 55.0,
        "coverage_level": 0.90,
        "method": "split_conformal"
      }
    },
    "120m": {
      "horizon_minutes": 120,
      "predicted_census": 50.8,
      "congestion_state": "CRITICAL",
      "prediction_interval": {
        "lower_census": 41.0,
        "upper_census": 60.5,
        "coverage_level": 0.90,
        "method": "split_conformal"
      }
    }
  },
  "bottleneck_signals": {
    "occupancy_surge": false,
    "acuity_pressure": true,
    "outflow_deficit": false,
    "queue_velocity_ratio": 1.2
  },
  "non_clinical_disclaimer": "Department congestion forecasts represent operational capacity projections and should not be used for clinical triage."
}
```

---

### 4.4 Queue Health Scoring

#### `POST /queue-health`
Computes the standardized composite Queue Health Score ($0–100$) synthesizing active bed occupancy, arrival velocity, and high-acuity workload. Categorizes state into `HEALTHY`, `MODERATE`, `BUSY`, or `CRITICAL` with dominant factor attribution.

**Request Body (`QueueHealthRequest`)**:
```json
{
  "active_census": 35.0,
  "recent_arrivals_60m": 8.0,
  "high_acuity_ratio": 0.18,
  "capacity": 50.0,
  "arrival_reference": 10.0,
  "acuity_reference": 0.20
}
```

**Response `200 OK` (`QueueHealthResponse`)**:
```json
{
  "score": 38.5,
  "state": "MODERATE",
  "components": {
    "occupancy": 35.0,
    "arrival": 40.0,
    "acuity": 45.0
  },
  "weights": {
    "w_occupancy": 0.50,
    "w_arrival": 0.30,
    "w_acuity": 0.20
  },
  "dominant_factor": "high-acuity clinical workload",
  "summary": "ED flow status is MODERATE with a Queue Health Score of 38.5/100. Primary operational pressure driven by high-acuity clinical workload.",
  "non_clinical_disclaimer": "Queue Health Score is a non-clinical operational load index designed solely for administrative capacity monitoring."
}
```

**Validation**:
- If custom weights `w_occupancy + w_arrival + w_acuity != 1.0`, returns `400 Bad Request`.
- If `active_census < 0` or `high_acuity_ratio` outside `[0.0, 1.0]`, returns `422 Unprocessable Content`.

---

### 4.5 Operational What-If Scenario Simulation

#### `POST /simulate/what-if`
Executes discrete-time counterfactual simulation for operational stress testing:
- `discharge_acceleration`: Models effect of discharge process acceleration (+X% departure throughput).
- `capacity_reduction`: Models temporary loss of operational beds (overflow stress).
- `arrival_surge`: Models sudden presentation spikes distributed across initial time intervals.

**Request Body (`SimulationRequest`)**:
```json
{
  "scenario_type": "discharge_acceleration",
  "time_steps": ["2026-09-05T08:00:00", "2026-09-05T08:30:00", "2026-09-05T09:00:00"],
  "initial_census": 40.0,
  "arrivals": [6.0, 5.0],
  "departures": [4.0, 5.0],
  "high_acuity_ratio": 0.20,
  "acceleration_rate": 0.25
}
```

**Response `200 OK` (`SimulationResponse`)**:
```json
{
  "scenario_name": "Discharge Acceleration (+25.0%)",
  "scenario_type": "discharge_acceleration",
  "time_steps": ["2026-09-05T08:00:00", "2026-09-05T08:30:00", "2026-09-05T09:00:00"],
  "baseline_census": [40.0, 42.0, 42.0],
  "simulated_census": [40.0, 41.0, 39.75],
  "delta_census": [0.0, -1.0, -2.25],
  "simulated_departures": [5.0, 6.25],
  "baseline_queue_health": {
    "score": 42.0,
    "state": "MODERATE",
    "components": {"occupancy": 42.0, "arrival": 35.0, "acuity": 45.0},
    "weights": {"w_occupancy": 0.5, "w_arrival": 0.3, "w_acuity": 0.2},
    "dominant_factor": "active bed occupancy",
    "summary": "Baseline queue health is MODERATE.",
    "non_clinical_disclaimer": "Non-clinical operational notice."
  },
  "simulated_queue_health": {
    "score": 39.8,
    "state": "MODERATE",
    "components": {"occupancy": 39.8, "arrival": 35.0, "acuity": 45.0},
    "weights": {"w_occupancy": 0.5, "w_arrival": 0.3, "w_acuity": 0.2},
    "dominant_factor": "active bed occupancy",
    "summary": "Simulated queue health is MODERATE.",
    "non_clinical_disclaimer": "Non-clinical operational notice."
  },
  "queue_stability": {
    "state": "STABLE",
    "net_flow": -1.25,
    "max_census": 41.0,
    "description": "Queue remains stable under simulated operational parameters."
  },
  "non_causal_disclaimer": {
    "is_causal": false,
    "notice": "Simulation outputs are deterministic counterfactual scenarios, NOT causal waiting-time inferences."
  }
}
```

---

## 5. Error Response Standard

All errors adhere to standardized JSON payloads:

```json
{
  "error": "Unprocessable Content",
  "status_code": 422,
  "detail": [
    {
      "loc": ["body", "stay_id"],
      "msg": "Data leakage prohibited field(s) detected in request: ['stay_id']",
      "type": "value_error"
    }
  ]
}
```

| HTTP Status | Trigger Conditions |
| :--- | :--- |
| `400 Bad Request` | Invalid domain parameters (e.g. Queue Health weights sum != 1.0) |
| `404 Not Found` | Request to unmapped path or resource |
| `422 Unprocessable Content` | Validation failure (Pydantic schema, physiological bounds, data leakage fields) |
| `500 Internal Server Error` | Unexpected unhandled server exception (logged with stack trace) |
| `503 Service Unavailable` | Model artifact unconfigured or unavailable on local filesystem |

---

## 6. Running the API Server

### Local Development
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Production Mode
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```
