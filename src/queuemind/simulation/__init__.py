"""Operational simulation package for QueueMind what-if counterfactual intelligence.

Provides:
- QueueStability: Operational stability states (STABLE, STRAINED, UNSTABLE).
- BaselineTrajectory: Container for baseline simulation intervals.
- ScenarioResult: Standardized comparison container between baseline and simulation.
- evaluate_queue_stability: Net flow and census growth classifier.
- simulate_discharge_acceleration: Discharge throughput enhancement scenario (+X%).
- simulate_capacity_reduction: Bed constraint and queue spillover scenario.
- simulate_arrival_surge: Intake presentation surge shock scenario (+N patients).
"""

from queuemind.simulation.what_if import (
    WAITING_TIME_UNAVAILABLE_PAYLOAD,
    BaselineTrajectory,
    QueueStability,
    ScenarioResult,
    evaluate_queue_stability,
    simulate_arrival_surge,
    simulate_capacity_reduction,
    simulate_discharge_acceleration,
)

__all__ = [
    "QueueStability",
    "BaselineTrajectory",
    "ScenarioResult",
    "WAITING_TIME_UNAVAILABLE_PAYLOAD",
    "evaluate_queue_stability",
    "simulate_discharge_acceleration",
    "simulate_capacity_reduction",
    "simulate_arrival_surge",
]
