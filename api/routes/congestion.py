"""Route handler for department-level multi-horizon congestion forecasting."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_congestion_predictor
from api.schemas import (
    CongestionForecastRequest,
    CongestionForecastResponse,
    HorizonForecastSchema,
)
from queuemind.models.congestion import CongestionPredictor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Department Congestion Forecasting"])


@router.post(
    "/predict/congestion",
    response_model=CongestionForecastResponse,
    summary="Forecast Department Active Census (30m, 60m, 120m)",
    description=(
        "Projects future Emergency Department active patient headcount across "
        "+30m, +60m, and +120m tactical planning horizons. Provides situational "
        "congestion states and queue dynamics bottleneck signals."
    ),
)
def predict_congestion(
    request: CongestionForecastRequest,
    predictor: CongestionPredictor = Depends(get_congestion_predictor),
) -> CongestionForecastResponse:
    """Generate multi-horizon active census forecasts and flow indicators."""
    snapshot_data = request.model_dump(exclude={"coverage_level"})

    try:
        raw_result = predictor.predict_single(
            snapshot_features=snapshot_data,
            coverage_level=request.coverage_level,
        )
    except (ValueError, TypeError) as exc:
        logger.warning("Congestion forecasting error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    forecast_dict: dict[str, HorizonForecastSchema] = {}
    for key, val in raw_result.get("forecasts", {}).items():
        forecast_dict[key] = HorizonForecastSchema(
            horizon_minutes=val["horizon_minutes"],
            predicted_census=val["predicted_census"],
            prediction_interval=val.get("prediction_interval"),
        )

    return CongestionForecastResponse(
        current_active_census=raw_result["current_active_census"],
        forecasts=forecast_dict,
        congestion_state=raw_result["congestion_state"],
        bottleneck_indicators=raw_result["bottleneck_indicators"],
        model_name=raw_result.get("model_name", predictor.model_name),
        model_version=raw_result.get("model_version", predictor.model_version),
    )
