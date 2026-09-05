"""Pydantic v2 request and response schemas for QueueMind REST API."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from queuemind.models.train import PROHIBITED_FEATURE_COLUMNS

# ==============================================================================
# Health & Status Schemas
# ==============================================================================


class HealthResponse(BaseModel):
    """API health status and service readiness metadata."""

    status: str = Field(default="ok", description="Process health state.")
    service: str = Field(default="queuemind-api", description="Service name.")
    version: str = Field(default="0.1.0", description="Application version.")
    environment: str = Field(default="development", description="Runtime environment.")
    models: dict[str, str] = Field(
        default_factory=lambda: {
            "patient_flow": "unavailable",
            "congestion": "unavailable",
        },
        description="Model artifact readiness status ('ready' or 'unavailable').",
    )

    model_config = ConfigDict(extra="forbid")


# ==============================================================================
# Patient Flow Prediction Schemas
# ==============================================================================


class PatientFlowRequest(BaseModel):
    """Snapshot features for predicting remaining patient stay duration."""

    # Patient & Clinical features at decision snapshot
    acuity: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description=(
            "Emergency Severity Index triage acuity (1=most acute, 5=least acute)."
        ),
    )
    temperature: float | None = Field(
        default=None,
        ge=70.0,
        le=115.0,
        description="Body temperature in Fahrenheit.",
    )
    heartrate: float | None = Field(
        default=None,
        ge=20.0,
        le=300.0,
        description="Heart rate in beats per minute.",
    )
    resprate: float | None = Field(
        default=None,
        ge=4.0,
        le=70.0,
        description="Respiratory rate in breaths per minute.",
    )
    o2sat: float | None = Field(
        default=None,
        ge=50.0,
        le=100.0,
        description="Oxygen saturation percentage.",
    )
    sbp: float | None = Field(
        default=None,
        ge=40.0,
        le=300.0,
        description="Systolic blood pressure (mmHg).",
    )
    dbp: float | None = Field(
        default=None,
        ge=20.0,
        le=200.0,
        description="Diastolic blood pressure (mmHg).",
    )
    pain: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Self-reported pain score (0 to 10).",
    )
    gender: str | None = Field(
        default=None,
        description="Patient gender (e.g. 'M', 'F').",
    )

    # Temporal features
    arrival_hour: int | None = Field(
        default=None,
        ge=0,
        le=23,
        description="Hour of arrival in 24h format (0–23).",
    )
    arrival_dayofweek: int | None = Field(
        default=None,
        ge=0,
        le=6,
        description="Day of week (0=Monday, 6=Sunday).",
    )
    is_weekend: int | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Binary weekend flag (1 if Saturday or Sunday).",
    )
    elapsed_time_minutes: float | None = Field(
        default=None,
        ge=0.0,
        description="Minutes elapsed since ED arrival up to the snapshot.",
    )

    # Congestion features
    active_census: float | None = Field(
        default=None,
        ge=0.0,
        description="Department active patient census at snapshot.",
    )
    recent_arrivals_60m: float | None = Field(
        default=None,
        ge=0.0,
        description="Department presentations in preceding 60 minutes.",
    )
    recent_departures_60m: float | None = Field(
        default=None,
        ge=0.0,
        description="Department departures in preceding 60 minutes.",
    )
    flow_ratio_60m: float | None = Field(
        default=None,
        ge=0.0,
        description="Arrival to departure velocity ratio (Laplace-smoothed).",
    )
    net_flow_60m: float | None = Field(
        default=None,
        description=(
            "Net volume change (arrivals - departures) in preceding 60 minutes."
        ),
    )
    high_acuity_census: float | None = Field(
        default=None,
        ge=0.0,
        description="Count of active patients with ESI <= 2.",
    )
    high_acuity_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ratio of active patients with ESI <= 2.",
    )

    # Prediction parameters
    coverage_level: float = Field(
        default=0.90,
        gt=0.0,
        lt=1.0,
        description="Conformal prediction interval coverage level in (0, 1).",
    )
    return_explanation: bool = Field(
        default=False,
        description="Whether to include SHAP feature attribution breakdown.",
    )

    @model_validator(mode="before")
    @classmethod
    def check_prohibited_leakage_fields(cls, data: Any) -> Any:
        """Reject any request containing leakage-prone future or target fields."""
        if isinstance(data, dict):
            leaked = set(data.keys()).intersection(PROHIBITED_FEATURE_COLUMNS)
            if leaked:
                raise ValueError(
                    "Data leakage prohibited field(s) detected in request: "
                    f"{sorted(leaked)}"
                )
        return data

    def to_feature_dict(self) -> dict[str, Any]:
        """Convert valid features to a clean dictionary for model scoring."""
        raw = self.model_dump(exclude={"coverage_level", "return_explanation"})
        return {k: v for k, v in raw.items() if v is not None}

    model_config = ConfigDict(extra="forbid")


class PredictionIntervalSchema(BaseModel):
    """Calibrated conformal prediction interval bounds."""

    lower_minutes: float = Field(
        ...,
        ge=0.0,
        description="Lower duration bound in minutes (clipped at 0.0).",
    )
    upper_minutes: float = Field(
        ...,
        ge=0.0,
        description="Upper duration bound in minutes.",
    )
    coverage_level: float = Field(
        ...,
        description="Target marginal statistical coverage level.",
    )
    method: str = Field(
        default="split_conformal",
        description="Conformal interval calibration method.",
    )
    non_negative_enforced: bool = Field(
        default=True,
        description="Whether physical non-negativity bound is active.",
    )


class FeatureContributionSchema(BaseModel):
    """SHAP attribution for a single feature."""

    name: str = Field(..., description="Feature identifier.")
    value: Any = Field(..., description="Observed feature value.")
    attribution: float = Field(
        ...,
        description=(
            "Attribution in minutes (positive increases time, negative decreases)."
        ),
    )
    direction: str = Field(
        ...,
        description="'increases_time' or 'decreases_time'.",
    )
    rank: int = Field(..., ge=1, description="Importance rank order.")


class PatientFlowExplanationSchema(BaseModel):
    """SHAP local feature attribution breakdown."""

    prediction: float = Field(..., description="Model point prediction in minutes.")
    base_value: float = Field(
        ...,
        description="Baseline expected prediction across training population.",
    )
    features: list[FeatureContributionSchema] = Field(
        ...,
        description="Ranked feature attributions.",
    )
    interpretation: str = Field(
        default=(
            "SHAP values explain model behavior for the prediction; "
            "they do not establish causality."
        ),
        description="Non-causal interpretation guidance.",
    )


class PatientFlowResponse(BaseModel):
    """Structured response for patient-flow remaining duration prediction."""

    predicted_remaining_time_minutes: float = Field(
        ...,
        description="Point prediction of remaining journey duration in minutes.",
    )
    unit: str = Field(default="minutes", description="Measurement unit.")
    model_name: str = Field(..., description="Prediction model identifier.")
    model_version: str = Field(..., description="Model version.")
    prediction_interval: PredictionIntervalSchema | None = Field(
        default=None,
        description="Calibrated conformal prediction interval.",
    )
    explanation: PatientFlowExplanationSchema | None = Field(
        default=None,
        description="SHAP local feature explanation decomposition.",
    )
    non_causal_disclaimer: str = Field(
        default=(
            "Predictions represent statistical estimates of remaining stay duration "
            "and do not constitute clinical advice, care prioritization, or guarantees."
        ),
        description="Clinical safety notice.",
    )


# ==============================================================================
# Congestion Forecasting Schemas
# ==============================================================================


class CongestionForecastRequest(BaseModel):
    """Department snapshot features for multi-horizon active census forecasting."""

    current_active_census: int = Field(
        ...,
        ge=0,
        description="Observed active patient headcount in ED at snapshot.",
    )
    recent_arrivals_15m: float = Field(default=0.0, ge=0.0)
    recent_arrivals_30m: float = Field(default=0.0, ge=0.0)
    recent_arrivals_60m: float = Field(default=0.0, ge=0.0)
    recent_arrivals_120m: float = Field(default=0.0, ge=0.0)
    recent_departures_15m: float = Field(default=0.0, ge=0.0)
    recent_departures_30m: float = Field(default=0.0, ge=0.0)
    recent_departures_60m: float = Field(default=0.0, ge=0.0)
    recent_departures_120m: float = Field(default=0.0, ge=0.0)
    net_flow_15m: float = Field(default=0.0)
    net_flow_30m: float = Field(default=0.0)
    net_flow_60m: float = Field(default=0.0)
    flow_ratio_60m: float = Field(default=1.0, ge=0.0)
    high_acuity_census: float = Field(default=0.0, ge=0.0)
    high_acuity_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    hour_sin: float = Field(default=0.0, ge=-1.0, le=1.0)
    hour_cos: float = Field(default=1.0, ge=-1.0, le=1.0)
    dayofweek: int = Field(default=0, ge=0, le=6)
    is_weekend: int = Field(default=0, ge=0, le=1)
    coverage_level: float = Field(default=0.90, gt=0.0, lt=1.0)

    model_config = ConfigDict(extra="forbid")


class HorizonForecastSchema(BaseModel):
    """Forecast and uncertainty for a specific forward horizon."""

    horizon_minutes: int = Field(..., description="Forecast horizon in minutes.")
    predicted_census: float = Field(
        ...,
        ge=0.0,
        description="Forecasted active headcount (clipped at 0.0).",
    )
    prediction_interval: dict[str, Any] | None = Field(
        default=None,
        description="Conformal prediction interval for headcount.",
    )


class CongestionForecastResponse(BaseModel):
    """Multi-horizon congestion forecasts and operational flow indicators."""

    current_active_census: int = Field(
        ...,
        description="Observed active census at decision snapshot.",
    )
    forecasts: dict[str, HorizonForecastSchema] = Field(
        ...,
        description="Forecasts keyed by horizon ('30m', '60m', '120m').",
    )
    congestion_state: dict[str, Any] = Field(
        ...,
        description="Situational congestion tier classification.",
    )
    bottleneck_indicators: dict[str, Any] = Field(
        ...,
        description="Queue dynamics bottleneck indicators.",
    )
    model_name: str = Field(..., description="Congestion model identifier.")
    model_version: str = Field(..., description="Model version.")
    non_clinical_disclaimer: str = Field(
        default=(
            "Congestion forecasts project departmental patient headcounts, not "
            "clinical disease progression, vital deterioration, or care urgency."
        ),
        description="Operational notice.",
    )


# ==============================================================================
# Queue Health Schemas
# ==============================================================================


class QueueHealthRequest(BaseModel):
    """Operational load parameters for computing Queue Health Score."""

    active_census: float = Field(
        ...,
        ge=0.0,
        description="Active patient headcount in the department.",
    )
    recent_arrivals_60m: float = Field(
        ...,
        ge=0.0,
        description="Presentations in the preceding 60 minutes.",
    )
    high_acuity_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of active patients with triage ESI <= 2.",
    )
    capacity_reference: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional operational bed capacity reference.",
    )
    arrival_rate_reference: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional nominal arrival intake reference (patients/hr).",
    )
    high_acuity_reference: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Optional high-acuity tolerance threshold.",
    )
    w_congestion: float | None = Field(default=None, gt=0.0)
    w_arrivals: float | None = Field(default=None, gt=0.0)
    w_acuity: float | None = Field(default=None, gt=0.0)

    model_config = ConfigDict(extra="forbid")


class QueueHealthResponse(BaseModel):
    """Standardized Queue Health Score (0–100) and component decomposition."""

    score: float = Field(..., ge=0.0, le=100.0, description="Queue Health Score.")
    state: str = Field(
        ...,
        description=(
            "Operational health category ('HEALTHY', 'MODERATE', 'BUSY', 'CRITICAL')."
        ),
    )
    components: dict[str, float] = Field(
        ...,
        description="Decomposed pressure scores (0–100).",
    )
    weights: dict[str, float] = Field(..., description="Applied component weights.")
    dominant_factor: str = Field(..., description="Primary driver of queue pressure.")
    summary: str = Field(..., description="Human-readable administrative summary.")
    non_clinical_disclaimer: str = Field(
        ...,
        description="Non-clinical operational notice.",
    )


# ==============================================================================
# What-If Simulation Schemas
# ==============================================================================


class SimulationRequest(BaseModel):
    """Request payload for operational what-if scenario simulation."""

    scenario_type: Literal[
        "discharge_acceleration", "capacity_reduction", "arrival_surge"
    ] = Field(..., description="What-if counterfactual scenario category.")
    time_steps: list[str] = Field(
        ...,
        min_length=2,
        description=(
            "ISO-formatted timestamps defining simulation intervals "
            "(len = intervals + 1)."
        ),
    )
    initial_census: float = Field(
        ...,
        ge=0.0,
        description="Active patient census at t=0.",
    )
    arrivals: list[float] = Field(
        ...,
        min_length=1,
        description="Baseline arrival counts per interval.",
    )
    departures: list[float] = Field(
        ...,
        min_length=1,
        description="Baseline departure counts per interval.",
    )
    high_acuity_ratio: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Proportion of active patients with ESI <= 2.",
    )

    # Scenario-specific parameters
    acceleration_rate: float | None = Field(
        default=0.20,
        ge=0.0,
        description=(
            "Relative departure throughput acceleration "
            "(for discharge_acceleration)."
        ),
    )
    reduced_capacity: float | None = Field(
        default=None,
        gt=0.0,
        description="Lowered operational bed capacity (for capacity_reduction).",
    )
    additional_arrivals: int | None = Field(
        default=10,
        ge=0,
        description="Total additional patients presenting (for arrival_surge).",
    )
    surge_duration_steps: int | None = Field(
        default=2,
        ge=1,
        description="Number of initial intervals absorbing the surge.",
    )
    surge_acuity_ratio: float | None = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Workload acuity ratio of incoming surge patients.",
    )

    @model_validator(mode="after")
    def validate_scenario_requirements(self) -> SimulationRequest:
        """Validate required parameters per scenario type."""
        if len(self.arrivals) != len(self.departures):
            raise ValueError(
                "arrivals and departures lists must have identical lengths."
            )
        if len(self.time_steps) != len(self.arrivals) + 1:
            raise ValueError(
                f"time_steps must have length len(arrivals) + 1 "
                f"(got {len(self.time_steps)} vs {len(self.arrivals) + 1})."
            )
        if self.scenario_type == "capacity_reduction" and self.reduced_capacity is None:
            raise ValueError(
                "reduced_capacity must be specified when scenario_type is "
                "'capacity_reduction'."
            )
        return self

    model_config = ConfigDict(extra="forbid")


class SimulationResponse(BaseModel):
    """Structured result of what-if scenario counterfactual simulation."""

    scenario_name: str
    scenario_type: str
    time_steps: list[str]
    baseline_census: list[float]
    simulated_census: list[float]
    census_delta: list[float]
    baseline_arrivals: list[float]
    simulated_arrivals: list[float]
    baseline_departures: list[float]
    simulated_departures: list[float]
    peak_baseline_census: float
    peak_simulated_census: float
    peak_delta: float
    final_baseline_census: float
    final_simulated_census: float
    baseline_queue_health: dict[str, Any]
    simulated_queue_health: dict[str, Any]
    stability: str
    waiting_time_impact: dict[str, str]
    limitations: list[str]
