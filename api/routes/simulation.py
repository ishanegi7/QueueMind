"""Route handler for operational what-if scenario counterfactual simulation."""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status
import pandas as pd

from api.schemas import SimulationRequest, SimulationResponse
from queuemind.simulation.what_if import (
    BaselineTrajectory,
    simulate_arrival_surge,
    simulate_capacity_reduction,
    simulate_discharge_acceleration,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["What-If Operational Simulation"])


@router.post(
    "/simulate/what-if",
    response_model=SimulationResponse,
    summary="Run Operational What-If Scenario Simulation",
    description=(
        "Executes deterministic discrete-time flow simulation under operational "
        "counterfactuals: discharge acceleration (+X%), capacity constraint "
        "reduction, or arrival presentation surges. Strictly separates simulation "
        "from causal waiting-time inference."
    ),
)
def simulate_what_if_scenario(
    request: SimulationRequest,
) -> SimulationResponse:
    """Execute counterfactual scenario simulation and return flow comparisons."""
    try:
        ts_list = [pd.Timestamp(ts) for ts in request.time_steps]
        baseline = BaselineTrajectory(
            time_steps=ts_list,
            initial_census=request.initial_census,
            arrivals=request.arrivals,
            departures=request.departures,
            high_acuity_ratio=request.high_acuity_ratio,
        )
    except (ValueError, TypeError) as exc:
        logger.warning("Baseline trajectory construction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid baseline trajectory parameters: {exc}",
        ) from exc

    try:
        if request.scenario_type == "discharge_acceleration":
            accel = (
                request.acceleration_rate
                if request.acceleration_rate is not None
                else 0.20
            )
            scenario_res = simulate_discharge_acceleration(
                baseline=baseline,
                acceleration_rate=accel,
            )
        elif request.scenario_type == "capacity_reduction":
            if request.reduced_capacity is None:
                raise ValueError("reduced_capacity is required for capacity_reduction.")
            scenario_res = simulate_capacity_reduction(
                baseline=baseline,
                reduced_capacity=request.reduced_capacity,
            )
        elif request.scenario_type == "arrival_surge":
            add_arr = (
                request.additional_arrivals
                if request.additional_arrivals is not None
                else 10
            )
            dur = (
                request.surge_duration_steps
                if request.surge_duration_steps is not None
                else 2
            )
            acuity = (
                request.surge_acuity_ratio
                if request.surge_acuity_ratio is not None
                else 0.50
            )
            scenario_res = simulate_arrival_surge(
                baseline=baseline,
                additional_arrivals=add_arr,
                surge_duration_steps=dur,
                surge_acuity_ratio=acuity,
            )
        else:
            raise ValueError(f"Unknown scenario type: {request.scenario_type}")
    except ValueError as exc:
        logger.warning("Scenario simulation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return SimulationResponse(**scenario_res.to_dict())
