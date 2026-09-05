# QueueMind — Product & Technical Requirements Document (PRD + TRD)

**Version:** 1.0  
**Status:** Development Blueprint  
**Project:** QueueMind — AI-Powered Emergency Department Flow Intelligence  
**Primary Goal:** Build a serious AIML portfolio project for DevFest 2026 showcasing applied ML, time-series forecasting, explainable AI, simulation, API engineering, and a production-style dashboard.

---

# 1. Product Overview

## 1.1 Product Vision

QueueMind is an AI-powered emergency department (ED) flow intelligence system that uses publicly available, de-identified healthcare data to:

1. Estimate a patient's likely remaining ED journey time.
2. Forecast upcoming ED congestion.
3. Quantify overall queue health.
4. Explain the major factors behind predictions.
5. Simulate operational what-if scenarios.

QueueMind is an **operations/patient-flow intelligence system**, not a clinical decision-support or diagnostic system.

## 1.2 One-Line Product Pitch

> QueueMind predicts emergency department patient-flow delays, forecasts congestion, explains why delays are occurring, and lets users explore operational what-if scenarios.

## 1.3 Core Problem

Emergency departments experience changing patient volumes, acuity, arrival rates, discharge rates, and workflow patterns. Traditional dashboards mostly describe the current state.

QueueMind aims to answer:

> **What is likely to happen next, and what factors are driving the pressure?**

---

# 2. Goals and Non-Goals

## 2.1 Goals

- Build a reproducible ML pipeline using existing public data.
- Predict patient-flow duration/remaining journey time.
- Forecast future ED load/congestion.
- Build an explainable AI layer.
- Create a Queue Health Score.
- Build an operational what-if simulation.
- Expose predictions through a FastAPI backend.
- Build a professional web dashboard.
- Containerize and deploy the application.
- Document methodology, limitations, experiments, and architecture.
- Demonstrate responsible AI and avoid clinical claims.

## 2.2 Non-Goals

QueueMind will NOT:

- diagnose diseases;
- recommend treatments;
- make automated triage decisions;
- recommend medication;
- replace doctors or nurses;
- claim exact real-world waiting time when the dataset does not contain a direct label;
- use real patient-identifying information;
- be presented as a clinically validated production system.

---

# 3. Target Users

## Primary User — Hospital Operations / Flow Manager

Needs:

- current ED load;
- arrival/discharge pressure;
- congestion forecast;
- expected patient-flow delay;
- explanation of major contributors;
- operational scenario simulation.

## Secondary User — Clinical/Administrative Staff

Needs:

- high-level patient-flow visibility;
- identification of unusually delayed cases;
- trend and congestion monitoring.

## Future User — Patient-Facing View

A future simplified version could show an estimated journey range.

Example:

> Estimated ED journey: 1h 45m – 3h 05m

This must never be represented as a guaranteed medical waiting-time prediction.

---

# 4. Key Product Features

## 4.1 Patient Journey Prediction

### Input

Potential inputs:

- arrival time;
- day of week;
- hour;
- acuity;
- arrival transport;
- triage vitals;
- chief complaint/category;
- current elapsed time;
- current ED load;
- recent arrival rate;
- recent discharge rate.

### Output

- predicted remaining time;
- prediction interval;
- delay-risk category;
- Queue Health Score.

Example:

```text
Predicted remaining ED time: 137 minutes

Likely range:
98 – 189 minutes

Delay risk:
MEDIUM
```

---

# 5. Congestion Forecasting

QueueMind should estimate future ED pressure for configurable horizons such as:

- next 30 minutes;
- next 60 minutes;
- next 120 minutes.

Potential output:

```text
Current Queue Status: BUSY

Next 60 minutes:
Congestion probability: HIGH

Trend:
WORSENING
```

---

# 6. Queue Health Score

Create a product-level score:

```text
Queue Health Score = 0–100
```

Initial interpretation:

| Score | Status |
|---:|---|
| 0–30 | Healthy |
| 31–60 | Moderate |
| 61–80 | Busy |
| 81–100 | Critical |

The score should combine measurable operational signals such as:

- active patient load;
- arrival pressure;
- discharge pressure;
- acuity pressure;
- predicted delay.

The exact weighting must be documented and validated.

---

# 7. Explainable AI

Every important prediction should have an explanation.

Example:

```text
Why is predicted remaining time high?

1. High current ED load
2. Higher patient acuity
3. Evening arrival period
4. High recent arrival rate
5. Similar historical cases had longer journeys
```

Primary explainability technology:

**SHAP**

Required outputs:

- global feature importance;
- per-prediction feature contribution;
- top positive contributors;
- top negative contributors.

---

# 8. What-If Simulation

QueueMind should allow users to modify operational assumptions.

Example:

```text
Current:

Active patients: 37
Arrival rate: 8/hour
Discharge rate: 5/hour
Queue Health: 81
```

Scenario:

```text
Increase discharge capacity:
5/hour → 7/hour
```

Simulation output:

```text
Expected queue pressure: reduced
Estimated active patients: lower
Queue Health: improved
Congestion status: HIGH → MODERATE
```

This feature is intended for **scenario exploration**, not operational decision-making without validation.

---

# 9. Dataset Strategy

## Primary Dataset

**MIMIC-IV-ED**

Use the official PhysioNet release and follow its access requirements.

Potential information includes:

- ED stays;
- arrival/discharge timestamps;
- acuity;
- arrival transport;
- triage information;
- vital signs;
- chief complaint;
- disposition;
- other ED workflow information.

## Secondary / Optional Dataset

**MIMICEL-ED**

Use when useful for event-level patient-flow modeling.

## Development Dataset

Use the official MIMIC-IV-ED demo dataset during early pipeline development.

## Data Rules

- Do not commit raw datasets to GitHub.
- Do not expose patient identifiers.
- Do not create fake clinical data to inflate results.
- Keep data-processing scripts reproducible.
- Document dataset version and access conditions.
- Maintain a data dictionary.

---

# 10. ML Problem Definition

## Model A — Patient Remaining-Time Regression

Target:

```text
remaining_time_minutes
```

For a historical prediction snapshot:

```text
remaining_time =
outtime - snapshot_time
```

Only features available at the snapshot time may be used.

### Important

No future information may enter the feature set.

---

# 11. Model B — ED Load / Congestion Forecasting

Predict:

```text
future_active_patient_count
```

or classify:

```text
future_congestion = LOW / MODERATE / HIGH / CRITICAL
```

Forecast horizons:

- 30 minutes;
- 60 minutes;
- 120 minutes.

---

# 12. Model C — Delay Risk

A derived classification layer:

```text
LOW
MEDIUM
HIGH
```

Thresholds must be calculated from the training data and documented.

---

# 13. Model Development Strategy

Do not jump directly to a complex model.

Build progressively:

### Baselines

1. Mean prediction.
2. Median prediction.
3. Linear Regression.

### ML Models

4. Random Forest.
5. XGBoost.

### Optional Future Models

- LightGBM;
- temporal models;
- sequence models;
- probabilistic regression;
- quantile regression.

The first production candidate should be **XGBoost for tabular patient-flow prediction**, subject to actual validation results.

---

# 14. Evaluation Metrics

## Regression

Primary:

- MAE;
- RMSE.

Secondary:

- R²;
- median absolute error;
- percentile error.

## Prediction Intervals

Evaluate:

- coverage;
- interval width;
- calibration.

## Congestion Classification

Use:

- precision;
- recall;
- F1;
- ROC-AUC where appropriate;
- confusion matrix.

For an operational warning system, recall for high-congestion events should be examined carefully rather than optimizing only overall accuracy.

---

# 15. Data Leakage Prevention

This is a mandatory requirement.

Do NOT randomly split snapshots from the same patient journey into train/test.

Preferred approach:

```text
Earlier encounters
        ↓
TRAIN

Later encounters
        ↓
VALIDATION

Latest encounters
        ↓
TEST
```

All feature generation must respect the prediction timestamp.

Any feature that uses future information is prohibited.

---

# 16. Product User Flow

```text
                    QueueMind Dashboard
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
    Patient Flow       Congestion         Analytics
          |                 |                 |
          v                 v                 v
     Prediction         Forecast          Explainability
          |                 |                 |
          +-----------------+-----------------+
                            |
                            v
                     Queue Health Score
                            |
                            v
                     What-If Simulation
```

---

# 17. Technical Architecture

```text
                MIMIC-IV-ED / MIMICEL
                         |
                         v
                 Data Ingestion
                         |
                         v
                 Data Validation
                         |
                         v
                Feature Engineering
                         |
             +-----------+-----------+
             |                       |
             v                       v
     Patient Flow Model       Congestion Model
             |                       |
             +-----------+-----------+
                         |
                         v
                    Model Layer
                         |
              +----------+----------+
              |                     |
              v                     v
         SHAP Layer          Simulation Engine
              |                     |
              +----------+----------+
                         |
                         v
                     FastAPI
                         |
                         v
                    Next.js UI
                         |
                         v
                       User
```

---

# 18. Technology Stack

## Data Engineering

- Python
- Pandas
- NumPy
- DuckDB
- Parquet

## Machine Learning

- scikit-learn
- XGBoost
- SHAP
- MLflow (optional initially)

## Backend

- FastAPI
- Pydantic
- Uvicorn

## Database

Initial:

- PostgreSQL

Optional cloud analytics:

- Google BigQuery

## Frontend

- Next.js
- TypeScript
- Tailwind CSS
- Recharts

## DevOps

- Git
- GitHub
- Docker
- Docker Compose
- GitHub Actions

## Google Cloud / DevFest Integration

Where technically justified:

- Google Cloud Run
- BigQuery
- Firebase
- Gemini API

Do not add Google services only for branding.

---

# 19. Repository Structure

```text
queuemind/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_patient_flow.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_experiments.ipynb
│
├── src/
│   └── queuemind/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── loader.py
│       │   ├── cleaner.py
│       │   └── validator.py
│       │
│       ├── features/
│       │   ├── patient_features.py
│       │   ├── temporal_features.py
│       │   └── congestion_features.py
│       │
│       ├── models/
│       │   ├── baseline.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   └── evaluate.py
│       │
│       ├── explainability/
│       │   └── shap_explainer.py
│       │
│       └── simulation/
│           └── what_if.py
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── prediction.py
│       ├── congestion.py
│       └── simulation.py
│
├── frontend/
│   └── ...
│
├── models/
│   ├── patient_time/
│   └── congestion/
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_api.py
│   └── test_simulation.py
│
└── docs/
    ├── PRD_TRD.md
    ├── architecture.md
    ├── methodology.md
    ├── data_dictionary.md
    └── model_card.md
```

---

# 20. Data Schema

## ed_visits

```text
stay_id
subject_id
intime
outtime
acuity
arrival_transport
gender
disposition
chiefcomplaint
```

## vitals

```text
stay_id
charttime
temperature
heartrate
resprate
o2sat
sbp
dbp
pain
```

## patient_snapshots

```text
stay_id
snapshot_time
elapsed_time
acuity
active_patient_count
arrival_rate
discharge_rate
queue_health_score
remaining_time
```

Actual columns must be adapted to the available dataset schema.

---

# 21. Feature Engineering

## Temporal

- hour;
- day of week;
- weekend;
- month;
- elapsed time.

## Patient

- acuity;
- heart rate;
- oxygen saturation;
- blood pressure;
- respiratory rate;
- temperature;
- pain;
- arrival transport;
- complaint category.

## Queue

- active patients;
- arrivals in last 15 minutes;
- arrivals in last 30 minutes;
- arrivals in last 60 minutes;
- discharges in last 15 minutes;
- discharges in last 30 minutes;
- discharges in last 60 minutes;
- arrival rate;
- discharge rate;
- arrival/discharge ratio;
- queue growth rate;
- acuity pressure.

---

# 22. Queue Health Engine

Initial conceptual design:

```text
Queue Health =
weighted(
    patient_load,
    arrival_pressure,
    discharge_pressure,
    acuity_pressure,
    predicted_delay
)
```

Normalize to:

```text
0–100
```

The exact formula and weights must be:

1. documented;
2. reproducible;
3. tested;
4. validated against historical operational outcomes where possible.

---

# 23. API Requirements

## Health

```http
GET /health
```

## Patient Prediction

```http
POST /predict/patient
```

Example request:

```json
{
  "acuity": 3,
  "heart_rate": 92,
  "oxygen_saturation": 98,
  "arrival_transport": "WALK IN",
  "hour": 18,
  "active_patients": 32
}
```

Example response:

```json
{
  "predicted_remaining_minutes": 137,
  "lower_bound": 98,
  "upper_bound": 189,
  "risk": "MEDIUM",
  "queue_health_score": 67
}
```

## Congestion

```http
GET /congestion/current
GET /congestion/forecast
```

## Simulation

```http
POST /simulate
```

## Model Metrics

```http
GET /model/metrics
```

---

# 24. API Engineering Requirements

- Pydantic validation;
- typed request/response schemas;
- structured errors;
- model loading at startup;
- no model training during inference;
- configurable model paths;
- health endpoint;
- logging;
- unit tests;
- API documentation.

---

# 25. Frontend Requirements

## Dashboard

Route:

```text
/dashboard
```

Show:

- Queue Health;
- active patients;
- arrival rate;
- discharge rate;
- congestion forecast;
- patient prediction;
- prediction explanation.

## Analytics

Route:

```text
/analytics
```

Show:

- patient-flow distributions;
- arrival trends;
- discharge trends;
- model metrics;
- feature importance.

## Simulation

Route:

```text
/simulation
```

Controls:

- arrival rate;
- discharge capacity;
- active patients;
- optional acuity pressure.

Output:

- projected queue health;
- projected load;
- congestion category;
- before/after comparison.

---

# 26. UI/UX Requirements

Design principles:

- professional;
- minimal;
- operations-dashboard aesthetic;
- accessible;
- responsive;
- readable charts;
- clear units;
- clear uncertainty;
- no misleading medical claims.

Avoid:

- excessive gradients;
- unnecessary animations;
- fake AI branding;
- meaningless counters;
- "99% accuracy" claims;
- decorative charts without operational meaning.

---

# 27. Vibe-Coding Development Rules

AI coding tools may be used for implementation, but every generated component must be reviewed and tested.

Core workflow:

```text
Plan
 ↓
Ask coding agent for ONE task
 ↓
Review generated changes
 ↓
Run tests
 ↓
Run application
 ↓
Inspect output
 ↓
Commit
```

Never ask the coding agent to build the entire project in one prompt.

---

# 28. Recommended Coding-Agent Workflow

## First Prompt

```text
You are a senior ML engineer helping me build QueueMind.

Read:
- docs/PRD_TRD.md

Do not implement the whole application.

First:
1. inspect the repository;
2. identify missing components;
3. propose milestones;
4. identify data and ML risks;
5. propose testing strategy.

Do not create fake clinical data.
Do not build the frontend yet.
Do not train models yet.

Wait for the next task.
```

## Data Loader Prompt

```text
Implement only the MIMIC-IV-ED data ingestion layer.

Requirements:
- Python
- pandas
- pathlib
- configurable input directory
- no hardcoded absolute paths
- type hints
- logging
- clear errors
- unit tests

Create:
src/queuemind/data/loader.py
tests/test_data.py

Do not build ML, API, or frontend code.
```

## Validation Prompt

```text
Implement the QueueMind data validation layer.

Check:
- required columns
- duplicate stay_id
- invalid timestamps
- outtime before intime
- missing acuity
- impossible vital values
- categorical inconsistencies

Return a validation report.

Do not modify the source dataset.
Add unit tests.
```

## Feature Pipeline Prompt

```text
Build the patient journey feature pipeline.

Create an encounter-level dataset from the available MIMIC-IV-ED tables.

Include:
- temporal features
- acuity
- arrival transport
- triage/vital features where available
- chief complaint/category where appropriate
- total ED duration for historical target construction

Strict requirement:
No target leakage.

Add tests and save processed data as Parquet.
```

## Model Prompt

```text
Implement the first QueueMind patient-flow prediction experiment.

Models:
1. Mean baseline
2. Median baseline
3. Linear Regression
4. Random Forest
5. XGBoost

Use a time-based train/validation/test split.

Metrics:
- MAE
- RMSE
- R2

Save:
- model
- feature list
- metrics JSON
- experiment configuration

Do not build the API or frontend.
```

---

# 29. Development Milestones

## V0.1 — Data Foundation

```text
MIMIC-IV-ED
 ↓
Loader
 ↓
Validation
 ↓
Clean dataset
 ↓
Data dictionary
```

## V0.2 — EDA

Analyze:

- ED duration;
- acuity;
- arrival patterns;
- discharge patterns;
- time-of-day effects;
- weekday/weekend effects.

## V0.3 — Patient Prediction

```text
Features
 ↓
Baselines
 ↓
Random Forest
 ↓
XGBoost
 ↓
Evaluation
```

## V0.4 — Explainability

```text
XGBoost
 ↓
SHAP
 ↓
Prediction explanation
```

## V0.5 — Congestion

```text
Historical event data
 ↓
Arrival/discharge features
 ↓
Forecast
 ↓
Congestion classification
```

## V0.6 — Queue Health

```text
Operational signals
 ↓
Queue Health Engine
 ↓
0–100 score
```

## V0.7 — Simulation

```text
Current state
 ↓
Scenario changes
 ↓
Simulation
 ↓
Projected queue state
```

## V0.8 — Backend

```text
Models
 ↓
FastAPI
 ↓
Validated endpoints
```

## V0.9 — Frontend

```text
Next.js
 ↓
Dashboard
 ↓
Analytics
 ↓
Simulation
```

## V1.0 — Production-style MVP

```text
Tests
+
Docker
+
CI
+
Documentation
+
Deployment
```

## V1.1 — DevFest Edition

Potential additions:

- Google Cloud Run;
- BigQuery;
- Firebase;
- Gemini-generated operational summaries;
- architecture diagram;
- demo video;
- technical blog;
- polished README;
- model card;
- responsible-AI documentation.

---

# 30. Seven-Day Initial Sprint

## Day 1

- create repository;
- create PRD_TRD.md;
- create project structure;
- create virtual environment;
- configure dependencies;
- configure Git;
- configure .gitignore.

## Day 2

- obtain authorized MIMIC-IV-ED demo/full data;
- inspect schema;
- build loader;
- build validation.

## Day 3

- create EDA notebook;
- analyze ED duration;
- analyze acuity;
- analyze temporal patterns.

## Day 4

- build feature pipeline;
- create snapshot logic;
- verify no leakage.

## Day 5

- build baseline models;
- build evaluation pipeline.

## Day 6

- train XGBoost;
- compare metrics;
- save experiment artifacts.

## Day 7

- build first prediction interface/CLI;
- add SHAP prototype;
- document findings;
- commit a clean milestone.

---

# 31. Git Strategy

Use small, meaningful commits.

Examples:

```text
chore: initialize QueueMind project
docs: add product and technical requirements
feat: add MIMIC data loader
feat: add data validation pipeline
feat: build patient journey features
feat: add baseline prediction models
feat: add XGBoost patient flow model
feat: add SHAP explanations
feat: add congestion forecasting
feat: add queue health scoring
feat: add what-if simulation
feat: add FastAPI prediction service
feat: add QueueMind dashboard
test: expand model and API coverage
docs: add architecture and methodology
chore: containerize application
```

---

# 32. Testing Requirements

## Unit Tests

- data loader;
- validators;
- feature functions;
- Queue Health calculation;
- simulation engine;
- model prediction interface.

## Integration Tests

- data → features;
- model → API;
- API → frontend.

## Regression Tests

After model changes:

- compare MAE;
- compare RMSE;
- verify feature schema;
- verify prediction distribution.

---

# 33. Responsible AI Requirements

QueueMind must clearly state:

- data is de-identified;
- results are retrospective;
- model is not clinically validated;
- predictions are estimates;
- uncertainty must be displayed;
- dataset biases may affect performance;
- performance may vary across populations and settings;
- model should not be used for clinical decisions.

---

# 34. Model Card Requirements

Create:

```text
docs/model_card.md
```

Include:

1. Model purpose.
2. Intended users.
3. Dataset.
4. Features.
5. Target.
6. Training methodology.
7. Evaluation.
8. Known limitations.
9. Bias considerations.
10. Out-of-scope use.
11. Reproducibility instructions.

---

# 35. Security & Privacy

- Never commit secrets.
- Use `.env`.
- Provide `.env.example`.
- Never commit raw healthcare data.
- Never expose patient identifiers.
- Validate API input.
- Limit logs to non-sensitive information.
- Do not store unnecessary patient-level data in production.

---

# 36. Deployment Architecture

Potential production-style architecture:

```text
                 Next.js Frontend
                         |
                         v
                    Cloud Run
                         |
                         v
                   FastAPI API
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Model Artifacts          PostgreSQL
             |
             v
         Prediction
             |
             v
       Optional BigQuery
```

Gemini can optionally generate natural-language operational summaries from **aggregated, non-identifying metrics**, not raw patient records.

---

# 37. Success Criteria

QueueMind MVP is successful when:

### Data

- reproducible ingestion works;
- validation catches known issues;
- raw data is never committed.

### ML

- baseline models are established;
- XGBoost is compared fairly;
- time-based evaluation is used;
- leakage checks pass;
- uncertainty is evaluated.

### Explainability

- SHAP explanations work for individual predictions;
- global feature importance is available.

### Operations

- congestion forecast works;
- Queue Health Score works;
- what-if simulation works.

### Engineering

- FastAPI serves predictions;
- frontend consumes API;
- automated tests pass;
- Docker build works;
- project can be reproduced from documentation.

### Portfolio

- README clearly explains the problem;
- architecture diagram exists;
- demo exists;
- model card exists;
- limitations are documented;
- Git history shows incremental engineering work.

---

# 38. Final Product Definition

QueueMind should ultimately demonstrate the following pipeline:

```text
PUBLIC ED DATA
      ↓
DATA ENGINEERING
      ↓
PATIENT JOURNEY REPRESENTATION
      ↓
FEATURE ENGINEERING
      ↓
MACHINE LEARNING
      ↓
PATIENT-FLOW PREDICTION
      ↓
CONGESTION FORECASTING
      ↓
EXPLAINABLE AI
      ↓
QUEUE HEALTH SCORE
      ↓
WHAT-IF SIMULATION
      ↓
FASTAPI
      ↓
PROFESSIONAL DASHBOARD
      ↓
CLOUD DEPLOYMENT
```

## Final Portfolio Positioning

> **QueueMind is an applied AI system for emergency department flow intelligence, combining tabular machine learning, temporal forecasting, explainable AI, and operational simulation to predict patient-flow delays and anticipate congestion.**

The project should be presented as a **research/engineering prototype**, not a clinically validated healthcare product.
