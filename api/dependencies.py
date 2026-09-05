"""Dependency injection providers for QueueMind FastAPI application."""

from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path

from fastapi import Depends, HTTPException, status

from api.config import Settings, get_settings
from queuemind.models.congestion import CongestionPredictor
from queuemind.models.predict import PatientFlowPredictor, load_predictor
from queuemind.queue_health.score import DEFAULT_HEALTH_CONFIG, QueueHealthConfig

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_cached_patient_flow_predictor(
    model_path_str: str,
) -> PatientFlowPredictor:
    """Load and cache PatientFlowPredictor instance from disk."""
    path = Path(model_path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found at: {path}")
    logger.info("Loading PatientFlowPredictor from %s", path)
    return load_predictor(path)


@lru_cache(maxsize=1)
def _load_cached_congestion_predictor(
    model_path_str: str,
) -> CongestionPredictor:
    """Load and cache CongestionPredictor instance from disk."""
    path = Path(model_path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Congestion model file not found at: {path}")
    logger.info("Loading CongestionPredictor from %s", path)
    return CongestionPredictor.load(path)


def get_patient_flow_predictor(
    settings: Settings = Depends(get_settings),
) -> PatientFlowPredictor:
    """Dependency returning the active PatientFlowPredictor or raising HTTP 503."""
    path_str = settings.PATIENT_FLOW_MODEL_PATH
    if not path_str:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Patient-flow prediction model artifact is not configured. "
                "Set PATIENT_FLOW_MODEL_PATH to a valid trained model artifact."
            ),
        )

    try:
        return _load_cached_patient_flow_predictor(path_str)
    except (FileNotFoundError, TypeError, Exception) as exc:
        logger.warning("Failed to load PatientFlowPredictor: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Patient-flow model artifact is unavailable at '{path_str}'. "
                "Ensure the model file exists and is valid."
            ),
        ) from exc


def get_congestion_predictor(
    settings: Settings = Depends(get_settings),
) -> CongestionPredictor:
    """Dependency returning the active CongestionPredictor or raising HTTP 503."""
    path_str = settings.CONGESTION_MODEL_PATH
    if not path_str:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Congestion forecasting model artifact is not configured. "
                "Set CONGESTION_MODEL_PATH to a valid trained model artifact."
            ),
        )

    try:
        return _load_cached_congestion_predictor(path_str)
    except (FileNotFoundError, TypeError, Exception) as exc:
        logger.warning("Failed to load CongestionPredictor: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Congestion forecasting artifact is unavailable at '{path_str}'. "
                "Ensure the model file exists and is valid."
            ),
        ) from exc


def get_queue_health_config() -> QueueHealthConfig:
    """Dependency providing the active QueueHealthConfig instance."""
    return DEFAULT_HEALTH_CONFIG
