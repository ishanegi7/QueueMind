"""QueueMind modeling package for emergency department patient-flow intelligence.

Provides:
- Baselines: GlobalMedianBaseline, AcuityStratifiedMedianBaseline,
  RidgeRegressionBaseline
- Training & Splitting: chronological_split, get_model_feature_names,
  build_preprocessor, XGBoostCandidate, PROHIBITED_FEATURE_COLUMNS
- Inference & Serialization: PatientFlowPredictor, save_predictor, load_predictor
- Evaluation: evaluate_regression, evaluate_subgroups, format_metrics_summary
"""

from queuemind.models.baseline import (
    AcuityStratifiedMedianBaseline,
    GlobalMedianBaseline,
    RidgeRegressionBaseline,
)
from queuemind.models.conformal import ConformalIntervalCalibrator
from queuemind.models.congestion import (
    PROHIBITED_CONGESTION_COLUMNS,
    CongestionPredictor,
    LastValueCongestionBaseline,
    RidgeCongestionModel,
    TimeOfDayMedianCongestionBaseline,
    XGBoostCongestionModel,
    filter_congestion_features,
    temporal_congestion_split,
)
from queuemind.models.evaluate import (
    evaluate_congestion_forecasts,
    evaluate_interval_subgroups,
    evaluate_prediction_intervals,
    evaluate_regression,
    evaluate_subgroups,
    format_congestion_metrics_summary,
    format_metrics_summary,
)
from queuemind.models.predict import (
    PatientFlowPredictor,
    load_predictor,
    save_predictor,
)
from queuemind.models.train import (
    PROHIBITED_FEATURE_COLUMNS,
    XGBoostCandidate,
    build_preprocessor,
    chronological_split,
    get_model_feature_names,
)

__all__ = [
    "AcuityStratifiedMedianBaseline",
    "GlobalMedianBaseline",
    "RidgeRegressionBaseline",
    "chronological_split",
    "get_model_feature_names",
    "build_preprocessor",
    "XGBoostCandidate",
    "PROHIBITED_FEATURE_COLUMNS",
    "PatientFlowPredictor",
    "save_predictor",
    "load_predictor",
    "evaluate_regression",
    "evaluate_subgroups",
    "format_metrics_summary",
    "ConformalIntervalCalibrator",
    "evaluate_prediction_intervals",
    "evaluate_interval_subgroups",
    "PROHIBITED_CONGESTION_COLUMNS",
    "temporal_congestion_split",
    "filter_congestion_features",
    "LastValueCongestionBaseline",
    "TimeOfDayMedianCongestionBaseline",
    "RidgeCongestionModel",
    "XGBoostCongestionModel",
    "CongestionPredictor",
    "evaluate_congestion_forecasts",
    "format_congestion_metrics_summary",
]
