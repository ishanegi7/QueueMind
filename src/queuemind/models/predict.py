"""Inference and model serialization module for QueueMind patient-flow forecasting.

This module encapsulates:
1. PatientFlowPredictor: Production inference wrapper bundling model artifacts,
   feature contracts, version metadata, and prediction methods.
2. Safe single-patient snapshot inference returning structured payloads.
3. Serialization utilities to persist and load model artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from queuemind.models.train import PROHIBITED_FEATURE_COLUMNS


class PatientFlowPredictor:
    """Production inference container for ED patient-flow remaining time prediction."""

    def __init__(
        self,
        model: Any,
        model_name: str,
        model_version: str = "0.1.0",
        feature_names: list[str] | None = None,
        numeric_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        target_col: str = "remaining_time_minutes",
        training_timestamp: str | None = None,
        calibrator: Any | None = None,
        explainer: Any | None = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.feature_names = list(feature_names) if feature_names is not None else []
        self.numeric_cols = list(numeric_cols) if numeric_cols is not None else []
        self.categorical_cols = (
            list(categorical_cols) if categorical_cols is not None else []
        )
        self.target_col = target_col
        self.training_timestamp = (
            training_timestamp or datetime.now(timezone.utc).isoformat()
        )
        self.calibrator = calibrator
        self.explainer = explainer

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate batch remaining time predictions in minutes.

        Args:
            X: DataFrame containing snapshot features.

        Returns:
            1D numpy array of predicted remaining time in minutes.

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If prohibited columns or missing required features are detected.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        # Guard against prohibited leakage
        leaked_cols = set(X.columns).intersection(PROHIBITED_FEATURE_COLUMNS)
        if leaked_cols:
            raise ValueError(
                "Data leakage detected! Feature matrix contains "
                f"prohibited columns: {leaked_cols}"
            )

        # Check required features if specified
        if self.feature_names:
            missing_cols = [col for col in self.feature_names if col not in X.columns]
            if missing_cols:
                raise ValueError(
                    "Input DataFrame is missing required feature columns: "
                    f"{missing_cols}"
                )
            input_df = X[self.feature_names].copy()
        else:
            input_df = X.copy()

        if hasattr(self.model, "predict"):
            preds = self.model.predict(input_df)
        else:
            raise NotFittedError(
                "Underlying model does not implement a predict method."
            )

        return np.asarray(preds, dtype=float)

    def predict_single(
        self,
        features: dict[str, Any],
        coverage_level: float | None = None,
        return_explanation: bool = False,
    ) -> dict[str, Any]:
        """Generate prediction for a single patient snapshot dictionary.

        Args:
            features: Dictionary containing point-in-time snapshot features.
            coverage_level: Desired conformal interval coverage level (e.g. 0.90).
            return_explanation: If True and an explainer is configured, attach
                SHAP feature contributions.

        Returns:
            Structured dictionary with predicted minutes, prediction interval,
            model metadata, and optional feature explanations.
        """
        df = pd.DataFrame([features])
        preds = self.predict(df)
        predicted_minutes = float(preds[0])

        calibrator = getattr(self, "calibrator", None)
        if calibrator is not None and getattr(calibrator, "is_calibrated", False):
            interval_dict = calibrator.get_interval_for_prediction(
                predicted_minutes, coverage_level=coverage_level
            )
        else:
            interval_dict = None

        result: dict[str, Any] = {
            "predicted_remaining_time_minutes": predicted_minutes,
            "prediction_interval": interval_dict,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "features_used": (
                list(self.feature_names)
                if self.feature_names
                else list(features.keys())
            ),
        }

        explainer = getattr(self, "explainer", None)
        if return_explanation and explainer is not None:
            result["explanation"] = explainer.explain_single(features)

        return result


def save_predictor(predictor: PatientFlowPredictor, file_path: str | Path) -> Path:
    """Serialize and save a PatientFlowPredictor to disk.

    Args:
        predictor: The PatientFlowPredictor instance to save.
        file_path: Destination path on disk.

    Returns:
        The resolved Path where the predictor was saved.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, path)
    return path


def load_predictor(file_path: str | Path) -> PatientFlowPredictor:
    """Load a serialized PatientFlowPredictor from disk.

    Args:
        file_path: Path to serialized predictor.

    Returns:
        The loaded PatientFlowPredictor instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the loaded object is not a PatientFlowPredictor.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Predictor file not found at: {path}")

    loaded = joblib.load(path)
    if not isinstance(loaded, PatientFlowPredictor):
        raise TypeError(
            "Expected loaded object to be PatientFlowPredictor, "
            f"but got: {type(loaded)}"
        )

    return loaded
