"""Health check and readiness route handler."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from api.config import Settings, get_settings
from api.schemas import HealthResponse

router = APIRouter(tags=["Health & System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API Process Health & Model Readiness",
    description=(
        "Returns current API process status and readiness states for patient-flow "
        "and congestion forecasting models. Does not guarantee data freshness or "
        "clinical correctness."
    ),
)
def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Check process health and model artifact availability."""
    pf_ready = False
    if settings.PATIENT_FLOW_MODEL_PATH:
        pf_ready = Path(settings.PATIENT_FLOW_MODEL_PATH).is_file()

    cg_ready = False
    if settings.CONGESTION_MODEL_PATH:
        cg_ready = Path(settings.CONGESTION_MODEL_PATH).is_file()

    return HealthResponse(
        status="ok",
        service="queuemind-api",
        version="0.1.0",
        environment=settings.ENVIRONMENT,
        models={
            "patient_flow": "ready" if pf_ready else "unavailable",
            "congestion": "ready" if cg_ready else "unavailable",
        },
    )
