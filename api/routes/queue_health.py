"""Route handler for standardized Queue Health Score calculation."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_queue_health_config
from api.schemas import QueueHealthRequest, QueueHealthResponse
from queuemind.queue_health.score import QueueHealthConfig, calculate_queue_health_score

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Queue Health Scoring"])


@router.post(
    "/queue-health",
    response_model=QueueHealthResponse,
    summary="Compute Department Queue Health Score (0–100)",
    description=(
        "Computes standardized composite Queue Health Score synthesizing active bed "
        "occupancy, arrival velocity, and high-acuity clinical workload. "
        "Categorizes state into HEALTHY, MODERATE, BUSY, or CRITICAL with "
        "dominant factor attribution."
    ),
)
def compute_queue_health(
    request: QueueHealthRequest,
    base_config: QueueHealthConfig = Depends(get_queue_health_config),
) -> QueueHealthResponse:
    """Calculate operational queue health score and component breakdown."""
    # Construct config override if custom parameters provided
    cfg = base_config
    if any(
        v is not None
        for v in (
            request.capacity_reference,
            request.arrival_rate_reference,
            request.high_acuity_reference,
            request.w_congestion,
            request.w_arrivals,
            request.w_acuity,
        )
    ):
        try:
            cfg = QueueHealthConfig(
                w_congestion=request.w_congestion or base_config.w_congestion,
                w_arrivals=request.w_arrivals or base_config.w_arrivals,
                w_acuity=request.w_acuity or base_config.w_acuity,
                capacity_reference=(
                    request.capacity_reference or base_config.capacity_reference
                ),
                arrival_rate_reference=(
                    request.arrival_rate_reference or base_config.arrival_rate_reference
                ),
                high_acuity_reference=(
                    request.high_acuity_reference or base_config.high_acuity_reference
                ),
                state_thresholds=base_config.state_thresholds,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid queue health configuration: {exc}",
            ) from exc

    try:
        raw_result = calculate_queue_health_score(
            active_census=request.active_census,
            recent_arrivals_60m=request.recent_arrivals_60m,
            high_acuity_ratio=request.high_acuity_ratio,
            config=cfg,
        )
    except ValueError as exc:
        logger.warning("Queue health calculation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return QueueHealthResponse(
        score=raw_result["score"],
        state=raw_result["state"],
        components=raw_result["components"],
        weights=raw_result["weights"],
        dominant_factor=raw_result["dominant_factor"],
        summary=raw_result["summary"],
        non_clinical_disclaimer=raw_result["non_clinical_disclaimer"],
    )
