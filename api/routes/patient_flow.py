"""Route handler for single-patient remaining journey time prediction."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_patient_flow_predictor
from api.schemas import (
    FeatureContributionSchema,
    PatientFlowExplanationSchema,
    PatientFlowRequest,
    PatientFlowResponse,
    PredictionIntervalSchema,
)
from queuemind.models.predict import PatientFlowPredictor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Patient Flow Prediction"])


@router.post(
    "/predict/patient-flow",
    response_model=PatientFlowResponse,
    summary="Predict Patient Remaining Journey Time",
    description=(
        "Predicts remaining patient-flow duration in minutes for a patient at a "
        "given decision snapshot. Rejects any leakage-prone future information. "
        "Returns optional conformal prediction intervals and SHAP feature "
        "attribution explanations."
    ),
)
def predict_patient_flow(
    request: PatientFlowRequest,
    predictor: PatientFlowPredictor = Depends(get_patient_flow_predictor),
) -> PatientFlowResponse:
    """Score patient snapshot features and return remaining duration and uncertainty."""
    features = request.to_feature_dict()

    try:
        raw_result = predictor.predict_single(
            features=features,
            coverage_level=request.coverage_level,
            return_explanation=request.return_explanation,
        )
    except (ValueError, TypeError) as exc:
        logger.warning("Patient flow prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Format conformal interval if available
    interval_schema = None
    raw_interval = raw_result.get("prediction_interval")
    if raw_interval is not None:
        interval_schema = PredictionIntervalSchema(
            lower_minutes=raw_interval["lower_minutes"],
            upper_minutes=raw_interval["upper_minutes"],
            coverage_level=raw_interval["coverage_level"],
            method=raw_interval.get("method", "split_conformal"),
            non_negative_enforced=raw_interval.get("non_negative_enforced", True),
        )

    # Format explanation if available
    explanation_schema = None
    raw_exp = raw_result.get("explanation")
    if raw_exp is not None:
        feature_items = [
            FeatureContributionSchema(
                name=str(feat["name"]),
                value=feat["value"],
                attribution=float(feat.get("shap_value", feat.get("attribution", 0.0))),
                direction=str(feat.get("direction", "neutral")),
                rank=int(feat.get("rank", 1)),
            )
            for feat in raw_exp.get("features", [])
        ]
        explanation_schema = PatientFlowExplanationSchema(
            prediction=raw_exp["prediction"],
            base_value=raw_exp["base_value"],
            features=feature_items,
        )

    return PatientFlowResponse(
        predicted_remaining_time_minutes=round(
            float(raw_result["predicted_remaining_time_minutes"]), 2
        ),
        unit="minutes",
        model_name=raw_result.get("model_name", predictor.model_name),
        model_version=raw_result.get("model_version", predictor.model_version),
        prediction_interval=interval_schema,
        explanation=explanation_schema,
    )
