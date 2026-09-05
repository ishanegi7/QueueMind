# QueueMind System Architecture

## 1. System Overview

QueueMind is an **AI-powered Emergency Department (ED) Patient Flow Intelligence System**. It provides operational visibility, patient journey duration forecasting, congestion alerts, and what-if capacity simulation using de-identified real-world datasets (primarily MIMIC-IV-ED).

```
                      +------------------------------------------+
                      |         Data Source: MIMIC-IV-ED         |
                      |  (PhysioNet Credentialed Access - DUA)   |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |          Data Ingestion Layer            |
                      |   (loader.py: Parquet, CSV.GZ, CSV)      |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |          Data Validation Layer           |
                      |  (validator.py: Schema, Bounds, Times)   |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |         Feature Engineering Store        |
                      |   Patient / Temporal / Queue Pressure    |
                      +------------------------------------------+
                                           |
                  +------------------------+------------------------+
                  |                                                 |
                  v                                                 v
+------------------------------------+            +------------------------------------+
|     Patient Flow Model Engine      |            |       Congestion Forecasting       |
| (Remaining Journey Time Regression)|            |   (Multi-Horizon Load & Pressure)  |
+------------------------------------+            +------------------------------------+
                  |                                                 |
                  +------------------------+------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |          Explainability Layer            |
                      |     (SHAP Feature Attributions)          |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |       Operational Simulation Engine      |
                      |      (Queue Health Score & What-If)      |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |            FastAPI Backend               |
                      |     (Type-Safe REST APIs via Pydantic)   |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |         Next.js Web Dashboard            |
                      |  (Tailwind CSS, Recharts, Interactive)   |
                      +------------------------------------------+
```

---

## 2. Core Architectural Components

### 2.1 Ingestion & Validation Layer (`src/queuemind/data/`)
- **Configurable Paths**: Accepts path inputs or reads `MIMIC_DATA_DIR` from environment variables without hardcoded platform-dependent paths.
- **Multi-Format Compatibility**: Seamlessly loads `.parquet`, `.csv.gz`, and `.csv`.
- **Targeted Table Loading**: Selectively reads only necessary tables (`edstays`, `diagnosis`, `medrecon`, `pyxis`, `triage`, `vitalsign`).
- **Strict Data Validation**: Validates schema conformity, non-emptiness, primary key uniqueness (`stay_id`), timestamp chronological coherence (`outtime >= intime`), and missing critical values prior to downstream consumption.

### 2.2 Feature Engineering Layer (`src/queuemind/features/`)
- **Temporal Features**: Hour of arrival, day of week, weekend indicator, seasonal trends, and elapsed ED stay duration.
- **Patient Attributes**: Acuity triage category (ESI 1–5), baseline vitals (heart rate, blood pressure, respiratory rate, O2 saturation, temperature, pain score), and chief complaint categorizations.
- **Queue State & Congestion Features**: Active patient counts, rolling arrival rates (15m, 30m, 60m windows), rolling discharge rates, arrival-to-discharge ratios, and acuity-weighted queue pressure.
- **Data Leakage Safeguards**: Strict timestamp cutoff alignment so that snapshot features incorporate only information generated up to the decision point.

### 2.3 Patient Flow Prediction Layer (`src/queuemind/models/`)
- **Baseline Models**: Mean, median, and Ridge/Linear regression models establishing benchmark performance floors.
- **Primary ML Models**: Gradient boosted decision trees (`XGBoostCandidate`) trained for remaining journey duration regression with scikit-learn preprocessing pipelines (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`).
- **Time-Based Evaluation**: Chronological train/validation/test partitioning preserving operational temporal ordering and eliminating lookahead bias.
- **Split-Conformal Prediction Calibration (`ConformalIntervalCalibrator`)**: Finite-sample distribution-free prediction intervals calibrated strictly on holdout chronological validation sets. Computes absolute residuals $R_i = |y_i - \hat{y}_i|$, finite-sample conformal cutoffs $q_{1-\alpha}$, and enforces physical non-negative duration bounds ($\max(0.0, \hat{y} - q)$).
- **Inference Packaging (`PatientFlowPredictor`)**: Production wrapper bundling preprocessing, model scoring, calibrated interval generation, and feature explanation in unified type-safe schemas.

### 2.4 Department Congestion Forecasting & Bottleneck Intelligence (`src/queuemind/models/congestion.py`)
- **Time-Grid Generator (`time_grid.py`)**: Generates regular snapshot grids and extracts department-level operational features strictly using information known $\le T$.
- **Direct Multi-Horizon Forecasting**: Separate direct models for $h \in \{30\text{m}, 60\text{m}, 120\text{m}\}$ predicting future physical active census.
- **Operational Benchmarks**:
  - `LastValueCongestionBaseline`: Persistence benchmark floor ($y(T+h) = \text{census}(T)$).
  - `TimeOfDayMedianCongestionBaseline`: Historical diurnal median baseline grouped by `(day_of_week, hour)`.
- **ML Models**: `RidgeCongestionModel` and `XGBoostCongestionModel` with non-negative headcount projection ($\ge 0$).
- **Temporal Embargo Protocol**: Chronological train/validation/test split enforcing a $\ge 120\text{m}$ embargo buffer between splits to eliminate target-overlap leakage.
- **Operational Flow Indicators (`bottleneck_features.py`)**: Evaluates mathematical queue dynamics signals (rising census velocity, high arrival pressure, low departure throughput, sustained positive net flow, acuity concentration) and classifies situational congestion state (`HEALTHY`, `MODERATE`, `BUSY`, `CRITICAL`).
- **Inference Packaging (`CongestionPredictor`)**: Production container providing multi-horizon forecasts, calibrated conformal bounds per horizon, and real-time bottleneck alerts.

### 2.5 Explainability Layer (`src/queuemind/explainability/`)
- **TreeSHAP Explanations (`ShapExplainer`)**: Fast tree-based Shapley value computation using Lundberg et al. (2020) TreeSHAP algorithm.
- **Feature Aggregation**: Mathematically consistent grouping of one-hot encoded indicator dummies back to clinical variables ($\phi_{\text{feature}} = \sum \phi_{\text{dummy}}$), preserving exact Shapley additivity ($\text{base\_value} + \sum \phi_i \approx \hat{y}$).
- **Directional Categorization & Ranking**: Categorizes feature contributions into flow accelerators (`increases_time`) versus expeditors (`decreases_time`) with rank ordering.
- **Strict Non-Causal Framing**: Explicitly documents and enforces that SHAP attributions represent local statistical associations within the trained model, not counterfactual or clinical causal mechanisms. Leakage-prohibited features are strictly rejected at the explainability boundary.

### 2.6 Simulation Layer (`src/queuemind/queue_health/` and `src/queuemind/simulation/`)
- **Queue Health Score (`queue_health/score.py`)**: Standardized 0–100 operational index synthesizing active bed occupancy ($w=0.50$), arrival intake velocity ($w=0.30$), and high-acuity workload ($w=0.20$). Includes configurable state categorization (`HEALTHY`, `MODERATE`, `BUSY`, `CRITICAL`), dominant factor attribution, and non-clinical operational summaries.
- **State Classification (`queue_health/states.py`)**: Type-safe enum and boundary validation (`QueueHealthState`, `QueueHealthStateThresholds`, `classify_queue_health_state`).
- **Discrete-Time Flow Conservation (`simulation/what_if.py`)**: `BaselineTrajectory` preserving discrete-time flow dynamics ($C(t+\Delta t) = \max(0, C(t) + \text{Arr} - \text{Dep})$) with non-negative active census clamping.
- **What-If Scenario Simulation Engine**:
  - `simulate_discharge_acceleration`: Models $+X\%$ throughput acceleration and bed-load reduction.
  - `simulate_capacity_reduction`: Models constrained operational capacity and tracks peak overflow.
  - `simulate_arrival_surge`: Injects $+N$ presentations across $M$ intervals to model surge absorption and recovery.
- **Queue Stability Evaluator (`evaluate_queue_stability`)**: Evaluates queue momentum into `STABLE`, `STRAINED`, or `UNSTABLE`.
- **Non-Causal Guardrail (`WAITING_TIME_UNAVAILABLE_PAYLOAD`)**: Transparently discloses the absence of interventional counterfactuals in MIMIC-IV-ED, rejecting causal waiting-time reduction claims.

### 2.7 Serving & User Interface (`api/` and `frontend/`)
- **FastAPI API Gateway**: Exposes asynchronous REST endpoints with Pydantic request/response schemas.
- **Next.js Dashboard**: Clinical operations interface presenting real-time flow forecasts, SHAP waterfall plots, queue health gauges, and scenario sliders.
