"""Operational what-if simulation and counterfactual scenario testing engine.

Implements deterministic discrete-time flow simulations:
1. Scenario A: Discharge Acceleration (+X% departure throughput).
2. Scenario B: Capacity Reduction (Operational bed constraint and overflow tracking).
3. Scenario C: Arrival Surge Shocks (+N presentations over M minutes).
4. Queue Stability evaluation (STABLE, STRAINED, UNSTABLE).

NON-CAUSAL / OPERATIONAL BOUNDARY DISCLAIMER:
Scenario outputs represent mathematical simulations under stated assumptions.
They do NOT establish that a clinical intervention would cause the predicted change.
MIMIC-IV-ED lacks interventional counterfactual data; direct individual waiting-time
claims are not supported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
from typing import Any, Sequence

import pandas as pd

from queuemind.queue_health.score import (
    DEFAULT_HEALTH_CONFIG,
    QueueHealthConfig,
    calculate_queue_health_score,
)

logger = logging.getLogger(__name__)


class QueueStability(str, Enum):
    """Operational queue stability categories based on net flow and accumulation."""

    STABLE = "STABLE"
    STRAINED = "STRAINED"
    UNSTABLE = "UNSTABLE"


WAITING_TIME_UNAVAILABLE_PAYLOAD: dict[str, str] = {
    "status": "unavailable",
    "reason": (
        "Current dataset lacks interventional counterfactual timestamps. Direct causal "
        "waiting-time reduction claims are scientifically unsupported."
    ),
    "operational_proxy": (
        "Active census changes reflect aggregate department bed-load adjustments, "
        "not individual patient wait time guarantees."
    ),
}


@dataclass
class BaselineTrajectory:
    """Baseline operational trajectory for simulation.

    Attributes:
        time_steps: List of ordered Timestamp intervals.
        initial_census: Active census at t=0.
        arrivals: List of arrival counts per interval.
        departures: List of departure counts per interval.
        high_acuity_ratio: Proportion of active patients with ESI <= 2.
    """

    time_steps: list[pd.Timestamp]
    initial_census: float
    arrivals: list[float]
    departures: list[float]
    high_acuity_ratio: float = 0.20

    def __post_init__(self) -> None:
        if len(self.arrivals) != len(self.departures):
            raise ValueError(
                "arrivals and departures lists must have identical lengths."
            )
        if len(self.time_steps) != len(self.arrivals) + 1:
            raise ValueError(
                f"time_steps must have length len(arrivals) + 1 "
                f"(got {len(self.time_steps)} vs {len(self.arrivals) + 1})."
            )
        if self.initial_census < 0:
            raise ValueError("initial_census cannot be negative.")
        if not (0.0 <= self.high_acuity_ratio <= 1.0):
            raise ValueError("high_acuity_ratio must be in [0.0, 1.0].")

    @property
    def census_trajectory(self) -> list[float]:
        """Compute deterministic baseline active census trajectory."""
        census = [float(self.initial_census)]
        curr = float(self.initial_census)
        for arr, dep in zip(self.arrivals, self.departures):
            curr = max(0.0, curr + float(arr) - float(dep))
            census.append(round(curr, 2))
        return census


@dataclass
class ScenarioResult:
    """Structured result of an operational what-if scenario simulation.

    Attributes:
        scenario_name: Human-readable scenario title.
        scenario_type: Category identifier ('discharge_acceleration', 'surge', etc.).
        time_steps: List of timestamps for simulation trajectory.
        baseline_census: Active census trajectory under baseline.
        simulated_census: Active census trajectory under scenario.
        census_delta: Difference (simulated - baseline) at each time step.
        baseline_arrivals: Baseline arrivals per interval.
        simulated_arrivals: Simulated arrivals per interval.
        baseline_departures: Baseline departures per interval.
        simulated_departures: Simulated departures per interval.
        peak_baseline_census: Maximum census in baseline trajectory.
        peak_simulated_census: Maximum census in simulated trajectory.
        peak_delta: Maximum reduction or increase in census.
        final_baseline_census: Final census in baseline trajectory.
        final_simulated_census: Final census in simulated trajectory.
        baseline_queue_health: Queue Health Score breakdown under baseline.
        simulated_queue_health: Queue Health Score breakdown under scenario.
        stability: Operational stability classification (STABLE, STRAINED, UNSTABLE).
        waiting_time_impact: Notice regarding individual waiting time limitations.
        limitations: List of stated mathematical assumptions.
    """

    scenario_name: str
    scenario_type: str
    time_steps: list[pd.Timestamp]
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
    waiting_time_impact: dict[str, str] = field(
        default_factory=lambda: WAITING_TIME_UNAVAILABLE_PAYLOAD.copy()
    )
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert ScenarioResult to a serializable dictionary."""
        d = asdict(self)
        d["time_steps"] = [t.isoformat() for t in self.time_steps]
        return d


def evaluate_queue_stability(
    arrivals: Sequence[float],
    departures: Sequence[float],
    initial_census: float,
    final_census: float,
) -> QueueStability:
    """Evaluate operational queue stability from intake, discharge, and census growth.

    Mathematical Criteria:
    - Net flow = sum(arrivals) - sum(departures)
    - Census growth = final_census - initial_census
    - STABLE: Net flow <= 0 and Census growth <= 0 (queue is stable or clearing).
    - UNSTABLE: Net flow > 10.0 or final_census > 1.5 * initial_census (runaway queue).
    - STRAINED: Net flow > 0 but contained (elevated pressure, non-runaway).

    Args:
        arrivals: Sequence of interval arrivals.
        departures: Sequence of interval departures.
        initial_census: Census at t=0.
        final_census: Census at t=end.

    Returns:
        QueueStability enum value.
    """
    total_arr = sum(arrivals)
    total_dep = sum(departures)
    net_flow = total_arr - total_dep
    census_growth = final_census - initial_census

    if net_flow <= 0.0 and census_growth <= 0.0:
        return QueueStability.STABLE
    elif net_flow > 10.0 or (
        initial_census > 0 and final_census > 1.5 * initial_census
    ):
        return QueueStability.UNSTABLE
    else:
        return QueueStability.STRAINED


def simulate_discharge_acceleration(
    baseline: BaselineTrajectory,
    acceleration_rate: float = 0.20,
    config: QueueHealthConfig | None = None,
) -> ScenarioResult:
    """Simulate operational impact of accelerated discharge throughput (+X%).

    COUNTERFACTUAL ASSUMPTION:
    Discharge velocity is scaled by (1 + acceleration_rate) during the simulation
    horizon while arrival flow remains invariant.

    Args:
        baseline: Baseline trajectory of arrivals, departures, and census.
        acceleration_rate: Relative increase in departure rate (e.g. 0.20 for +20%).
        config: Optional QueueHealthConfig override.

    Returns:
        ScenarioResult comparing baseline and accelerated discharge trajectories.

    Raises:
        ValueError: If acceleration_rate is negative.
    """
    if acceleration_rate < 0.0:
        raise ValueError(
            f"acceleration_rate cannot be negative; got {acceleration_rate}"
        )

    cfg = config or DEFAULT_HEALTH_CONFIG
    base_census = baseline.census_trajectory

    # Apply discharge acceleration
    sim_dep = [round(dep * (1.0 + acceleration_rate), 2) for dep in baseline.departures]
    sim_arr = list(baseline.arrivals)

    # State transition: C(t+1) = max(0, C(t) + arr - dep)
    sim_census = [float(baseline.initial_census)]
    curr = float(baseline.initial_census)
    for arr, dep in zip(sim_arr, sim_dep):
        curr = max(0.0, curr + arr - dep)
        sim_census.append(round(curr, 2))

    delta = [round(s - b, 2) for s, b in zip(sim_census, base_census)]

    peak_base = max(base_census)
    peak_sim = max(sim_census)
    peak_reduction = round(peak_base - peak_sim, 2)

    final_base = base_census[-1]
    final_sim = sim_census[-1]

    # Queue Health comparison at final step
    rolling_base_arr = (
        sum(baseline.arrivals[-4:])
        if len(baseline.arrivals) >= 4
        else sum(baseline.arrivals)
    )
    base_qh = calculate_queue_health_score(
        active_census=final_base,
        recent_arrivals_60m=rolling_base_arr,
        high_acuity_ratio=baseline.high_acuity_ratio,
        config=cfg,
    )
    sim_qh = calculate_queue_health_score(
        active_census=final_sim,
        recent_arrivals_60m=rolling_base_arr,
        high_acuity_ratio=baseline.high_acuity_ratio,
        config=cfg,
    )

    stability = evaluate_queue_stability(
        arrivals=sim_arr,
        departures=sim_dep,
        initial_census=baseline.initial_census,
        final_census=final_sim,
    )

    limitations = [
        f"Assumes departure throughput can be accelerated by "
        f"{round(acceleration_rate * 100, 1)}% without clinical bottlenecks.",
        "Arrival volume is assumed invariant to discharge throughput.",
        "Does not claim real-world clinical feasibility.",
    ]

    return ScenarioResult(
        scenario_name=f"+{int(acceleration_rate * 100)}% Discharge Acceleration",
        scenario_type="discharge_acceleration",
        time_steps=baseline.time_steps,
        baseline_census=base_census,
        simulated_census=sim_census,
        census_delta=delta,
        baseline_arrivals=baseline.arrivals,
        simulated_arrivals=sim_arr,
        baseline_departures=baseline.departures,
        simulated_departures=sim_dep,
        peak_baseline_census=peak_base,
        peak_simulated_census=peak_sim,
        peak_delta=peak_reduction,
        final_baseline_census=final_base,
        final_simulated_census=final_sim,
        baseline_queue_health=base_qh,
        simulated_queue_health=sim_qh,
        stability=stability.value,
        limitations=limitations,
    )


def simulate_capacity_reduction(
    baseline: BaselineTrajectory,
    reduced_capacity: float,
    config: QueueHealthConfig | None = None,
) -> ScenarioResult:
    """Simulate operational impact when departmental capacity is reduced.

    OPERATIONAL ASSUMPTION:
    Bed capacity is an abstract configurable operational parameter.
    When active census exceeds reduced_capacity, queue overflow occurs.

    Args:
        baseline: Baseline trajectory.
        reduced_capacity: Lowered operational capacity threshold in patient beds.
        config: Optional QueueHealthConfig.

    Returns:
        ScenarioResult modeling queue strain under constricted capacity.

    Raises:
        ValueError: If reduced_capacity is non-positive.
    """
    if reduced_capacity <= 0.0:
        raise ValueError(f"reduced_capacity must be positive; got {reduced_capacity}")

    cfg = config or DEFAULT_HEALTH_CONFIG
    base_census = baseline.census_trajectory

    # Create config with reduced capacity reference
    constrained_cfg = QueueHealthConfig(
        w_congestion=cfg.w_congestion,
        w_arrivals=cfg.w_arrivals,
        w_acuity=cfg.w_acuity,
        capacity_reference=reduced_capacity,
        arrival_rate_reference=cfg.arrival_rate_reference,
        high_acuity_reference=cfg.high_acuity_reference,
        state_thresholds=cfg.state_thresholds,
    )

    # Physical census trajectory is unchanged; capacity pressure increases
    sim_census = list(base_census)
    delta = [0.0] * len(sim_census)

    peak_base = max(base_census)
    final_base = base_census[-1]
    peak_overflow = max(0.0, peak_base - reduced_capacity)

    rolling_arr = (
        sum(baseline.arrivals[-4:])
        if len(baseline.arrivals) >= 4
        else sum(baseline.arrivals)
    )

    base_qh = calculate_queue_health_score(
        active_census=final_base,
        recent_arrivals_60m=rolling_arr,
        high_acuity_ratio=baseline.high_acuity_ratio,
        config=cfg,
    )
    sim_qh = calculate_queue_health_score(
        active_census=final_base,
        recent_arrivals_60m=rolling_arr,
        high_acuity_ratio=baseline.high_acuity_ratio,
        config=constrained_cfg,
    )

    stability = (
        QueueStability.UNSTABLE
        if peak_overflow > 10.0
        else (QueueStability.STRAINED if peak_overflow > 0.0 else QueueStability.STABLE)
    )

    limitations = [
        f"Operational bed capacity is treated as an abstract parameter "
        f"({reduced_capacity} beds), not a historical physical bed count.",
        "Assumes staff-to-patient ratios and discharge processes remain unchanged.",
    ]

    return ScenarioResult(
        scenario_name=f"Capacity Constrained to {int(reduced_capacity)} Beds",
        scenario_type="capacity_reduction",
        time_steps=baseline.time_steps,
        baseline_census=base_census,
        simulated_census=sim_census,
        census_delta=delta,
        baseline_arrivals=baseline.arrivals,
        simulated_arrivals=baseline.arrivals,
        baseline_departures=baseline.departures,
        simulated_departures=baseline.departures,
        peak_baseline_census=peak_base,
        peak_simulated_census=peak_base,
        peak_delta=peak_overflow,
        final_baseline_census=final_base,
        final_simulated_census=final_base,
        baseline_queue_health=base_qh,
        simulated_queue_health=sim_qh,
        stability=stability.value,
        limitations=limitations,
    )


def simulate_arrival_surge(
    baseline: BaselineTrajectory,
    additional_arrivals: int = 10,
    surge_duration_steps: int = 2,
    surge_acuity_ratio: float = 0.50,
    config: QueueHealthConfig | None = None,
) -> ScenarioResult:
    """Simulate an arrival surge shock (e.g. +N presentations over M intervals).

    OPERATIONAL ASSUMPTION:
    The additional presentations are distributed uniformly across the specified
    surge intervals starting from t=0. Subsequent steps observe recovery dynamics.

    Args:
        baseline: Baseline trajectory.
        additional_arrivals: Total additional patients presenting (must be >= 0).
        surge_duration_steps: Number of initial intervals absorbing the surge.
        surge_acuity_ratio: Workload acuity ratio of incoming surge patients.
        config: Optional QueueHealthConfig.

    Returns:
        ScenarioResult showing surge absorption and post-surge recovery.

    Raises:
        ValueError: If additional_arrivals < 0 or surge_duration_steps < 1.
    """
    if additional_arrivals < 0:
        raise ValueError(
            f"additional_arrivals cannot be negative; got {additional_arrivals}"
        )
    if surge_duration_steps < 1:
        raise ValueError(
            f"surge_duration_steps must be at least 1; got {surge_duration_steps}"
        )

    cfg = config or DEFAULT_HEALTH_CONFIG
    base_census = baseline.census_trajectory
    n_intervals = len(baseline.arrivals)

    effective_steps = min(surge_duration_steps, n_intervals)
    arrivals_per_step = additional_arrivals / float(effective_steps)

    sim_arr = list(baseline.arrivals)
    for i in range(effective_steps):
        sim_arr[i] = round(sim_arr[i] + arrivals_per_step, 2)

    sim_dep = list(baseline.departures)

    # State transition
    sim_census = [float(baseline.initial_census)]
    curr = float(baseline.initial_census)
    for arr, dep in zip(sim_arr, sim_dep):
        curr = max(0.0, curr + arr - dep)
        sim_census.append(round(curr, 2))

    delta = [round(s - b, 2) for s, b in zip(sim_census, base_census)]

    peak_base = max(base_census)
    peak_sim = max(sim_census)
    max_increase = round(peak_sim - peak_base, 2)

    final_base = base_census[-1]
    final_sim = sim_census[-1]

    rolling_arr = sum(sim_arr[-4:]) if len(sim_arr) >= 4 else sum(sim_arr)
    base_qh = calculate_queue_health_score(
        active_census=final_base,
        recent_arrivals_60m=sum(baseline.arrivals[-4:]),
        high_acuity_ratio=baseline.high_acuity_ratio,
        config=cfg,
    )
    sim_qh = calculate_queue_health_score(
        active_census=final_sim,
        recent_arrivals_60m=rolling_arr,
        high_acuity_ratio=surge_acuity_ratio,
        config=cfg,
    )

    stability = evaluate_queue_stability(
        arrivals=sim_arr,
        departures=sim_dep,
        initial_census=baseline.initial_census,
        final_census=final_sim,
    )

    limitations = [
        f"Simulates {additional_arrivals} additional presentations uniformly "
        f"distributed across {effective_steps} intervals.",
        "Discharge velocity is assumed constant (no automatic staffing escalation).",
        "Acuity concentration reflects surge workload, not clinical prognosis.",
    ]

    return ScenarioResult(
        scenario_name=f"+{additional_arrivals} Arrival Surge Shock",
        scenario_type="arrival_surge",
        time_steps=baseline.time_steps,
        baseline_census=base_census,
        simulated_census=sim_census,
        census_delta=delta,
        baseline_arrivals=baseline.arrivals,
        simulated_arrivals=sim_arr,
        baseline_departures=baseline.departures,
        simulated_departures=sim_dep,
        peak_baseline_census=peak_base,
        peak_simulated_census=peak_sim,
        peak_delta=max_increase,
        final_baseline_census=final_base,
        final_simulated_census=final_sim,
        baseline_queue_health=base_qh,
        simulated_queue_health=sim_qh,
        stability=stability.value,
        limitations=limitations,
    )
