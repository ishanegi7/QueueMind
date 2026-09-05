# QueueMind: Emergency Department Patient Flow Intelligence System

> 🚧 **Project Status: Currently in development — Data Foundation phase.**

[![CI Quality & Test Suite](https://github.com/queuemind/QueueMind/actions/workflows/ci.yml/badge.svg)](https://github.com/queuemind/QueueMind/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 1. What is QueueMind?

**QueueMind** is an open-source, AI-driven Emergency Department (ED) Patient Flow Intelligence System designed to help hospital operations teams anticipate bottlenecks, predict remaining patient journey durations, forecast departmental congestion, and simulate the operational impact of capacity adjustments.

### ⚠️ Strict Non-Goal & Clinical Disclaimer
> **QueueMind is an operational and patient-flow management platform, NOT a clinical diagnostic or triage system.**  
> It does not diagnose diseases, recommend medical treatments, or make automated clinical decisions. Its sole purpose is operational queue intelligence, flow optimization, and capacity visibility.

---

## 2. The Real-World Problem

Emergency Departments are dynamic, high-stress operational environments characterized by:
- Unpredictable surges in patient arrival rates and sudden shifts in clinical acuity.
- Bed-block and boarding delays when downstream inpatient beds are unavailable.
- Traditional ED dashboards that are purely retrospective—describing what has *already* occurred rather than forecasting what is likely to happen in the next 30, 60, or 120 minutes.

QueueMind bridges this gap by answering two vital operational questions:
1. **How much longer is a given patient likely to remain in the ED journey given current system load?**
2. **What will departmental congestion look like over upcoming operational horizons, and what factors are driving the pressure?**

---

## 3. Dataset & Privacy Governance

QueueMind relies exclusively on real-world, de-identified healthcare data from the credentialed **[MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/)** database (Beth Israel Deaconess Medical Center, Boston, MA), hosted by PhysioNet.

### Data Access & Compliance
- **Zero Real Data in Version Control**: Under the PhysioNet Data Use Agreement (DUA) and HIPAA regulations, no clinical datasets, de-identified patient tables, or derived PHI are stored in this repository.
- **No Automatic Downloading**: Users must be individually credentialed on PhysioNet, complete CITI human-subjects training, and sign the DUA before manually placing authorized data in `data/raw/`.
- **No Synthetic Healthcare Data**: We do not create, invent, or commit fabricated medical datasets.
- For setup instructions, see [data/README.md](data/README.md).

---

## 4. Key Capabilities (Architectural Vision)

- **Patient Journey Duration Prediction**: Gradient boosted regression (XGBoost) estimating remaining stay time with calibrated prediction intervals.
- **Congestion Forecasting**: Multi-horizon predictive classification (30, 60, and 120 minutes ahead) classifying queue stress levels.
- **Explainable AI (SHAP)**: Granular per-prediction Shapley value attributions explaining the operational drivers behind estimated delays.
- **Queue Health Score**: A composite 0–100 operational metric synthesizing patient census, acuity distribution, and flow velocity.
- **What-If Scenario Simulation**: Counterfactual evaluation allowing operations managers to model the effects of capacity changes (e.g., +20% discharge velocity).
- **FastAPI Backend & Next.js UI**: Production-grade type-safe API with an interactive operations dashboard.

---

## 5. Development Roadmap

| Phase | Milestone | Description | Status |
|---|---|---|:---:|
| **Phase 1** | **Data Foundation** | Repository scaffolding, MIMIC-IV-ED loader, schema validation, test framework, and documentation | ✅ **Completed** |
| **Phase 2** | **Feature Engineering** | Patient features, temporal dynamics, congestion pressure metrics, and point-in-time snapshot generator | 🔄 Planned |
| **Phase 3** | **Baseline & ML Models** | Benchmark regressors, XGBoost training, hyperparameter optimization, and chronological evaluation | 🔄 Planned |
| **Phase 4** | **Explainability & Simulation** | TreeSHAP explainer integration, Queue Health Score engine, and what-if simulation module | 🔄 Planned |
| **Phase 5** | **API & Interactive Dashboard** | FastAPI endpoints, Next.js frontend, Recharts visual analytics, and scenario explorer | 🔄 Planned |

---

## 6. Project Directory Structure

```text
QueueMind/
├── README.md                          # Project overview and instructions
├── LICENSE                            # MIT License
├── .gitignore                         # Data privacy & artifact ignore rules
├── .env.example                       # Environment configuration template
├── pyproject.toml                     # Python package & tool configuration
├── requirements.txt                   # Project dependencies
├── docs/                              # Architecture and methodology documentation
│   ├── PRD_TRD.md                     # Product & Technical Requirements Document
│   ├── architecture.md                # System architecture specification
│   ├── methodology.md                 # Analytical design & leakage prevention
│   ├── data_dictionary.md             # MIMIC-IV-ED schema reference
│   └── model_card.md                  # Responsible AI model documentation
├── data/
│   ├── raw/                           # Raw MIMIC-IV-ED tables (gitignored)
│   ├── processed/                     # Processed features (gitignored)
│   └── README.md                      # Data acquisition and compliance guide
├── notebooks/                         # Exploratory data analysis notebooks
├── src/
│   └── queuemind/
│       ├── __init__.py
│       ├── data/                      # Ingestion, validation, and cleaning
│       │   ├── __init__.py
│       │   ├── loader.py              # Multi-format MIMIC-IV-ED loader
│       │   ├── cleaner.py             # Data normalization & plausibility
│       │   └── validator.py           # Schema & integrity validation
│       ├── features/                  # Feature engineering modules
│       ├── models/                    # Model training, inference, and evaluation
│       ├── explainability/            # SHAP attribution components
│       └── simulation/                # What-if and Queue Health engines
├── api/                               # FastAPI gateway (Phase 5)
├── frontend/                          # Next.js web dashboard (Phase 5)
├── models/                            # Local model weights directory (gitignored)
├── tests/                             # Pytest test suite
│   ├── __init__.py
│   └── test_data.py                   # Data loader and validator unit tests
└── .github/
    └── workflows/
        └── ci.yml                     # Continuous Integration workflow
```

---

## 7. Quick Start (Development)

### Prerequisites
- Python 3.11 or higher
- Git

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/queuemind/QueueMind.git
cd QueueMind

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
```

### Running Tests and Quality Checks
```bash
# Run pytest with code coverage
pytest

# Run code style and type checkers
black --check src tests
flake8 src tests
mypy src/queuemind
```

---

## 8. License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
