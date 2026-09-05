"""Queue Health Score calculation and component decomposition.

Synthesizes three operational pressure vectors into a standardized index [0, 100]:
1. Congestion Pressure: Active patient count relative to capacity reference.
2. Arrival Pressure: Recent arrival intake rate relative to reference volume.
3. High-Acuity Concentration: Proportion of active beds occupied by critical ESI 1–2
   patients relative to baseline workload tolerance.

NON-CLINICAL DISCLAIMER:
Queue Health Scores quantify operational queueing pressure and workload demand.
They do NOT assess patient clinical risk, diagnostic urgency, or medical care quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

from queuemind.queue_health.states import (
    DEFAULT_HEALTH_THRESHOLDS,
    QueueHealthState,
    QueueHealthStateThresholds,
    classify_queue_health_state,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueueHealthConfig:
    """Configuration for Queue Health Score calculation and normalization.

    Attributes:
        w_congestion: Weight assigned to congestion pressure (default: 0.50).
        w_arrivals: Weight assigned to arrival intake pressure (default: 0.30).
        w_acuity: Weight assigned to high-acuity workload pressure (default: 0.20).
        capacity_reference: Baseline active census capacity level (default: 50.0).
        arrival_rate_reference: Hourly arrival volume reference level (default: 10.0).
        high_acuity_reference: High-acuity fraction reference level (default: 0.40).
        state_thresholds: Threshold configuration for operational state mapping.
    """

    w_congestion: float = 0.50
    w_arrivals: float = 0.30
    w_acuity: float = 0.20
    capacity_reference: float = 50.0
    arrival_rate_reference: float = 10.0
    high_acuity_reference: float = 0.40
    state_thresholds: QueueHealthStateThresholds = DEFAULT_HEALTH_THRESHOLDS

    def __post_init__(self) -> None:
        total_w = self.w_congestion + self.w_arrivals + self.w_acuity
        if not np.isclose(total_w, 1.0, atol=1e-5):
            raise ValueError(
                f"Component weights must sum to 1.0; got "
                f"{self.w_congestion} + {self.w_arrivals} + {self.w_acuity} = {total_w}"
            )

        if min(self.w_congestion, self.w_arrivals, self.w_acuity) <= 0.0:
            raise ValueError("All component weights must be strictly positive.")

        if self.capacity_reference <= 0.0:
            raise ValueError("capacity_reference must be positive.")

        if self.arrival_rate_reference <= 0.0:
            raise ValueError("arrival_rate_reference must be positive.")

        if not (0.0 < self.high_acuity_reference <= 1.0):
            raise ValueError("high_acuity_reference must be in (0.0, 1.0].")


DEFAULT_HEALTH_CONFIG = QueueHealthConfig()


def calculate_queue_health_score(
    active_census: float | int,
    recent_arrivals_60m: float | int,
    high_acuity_ratio: float,
    config: QueueHealthConfig | None = None,
) -> dict[str, Any]:
    """Calculate a standardized Queue Health Score (0–100) and component decomposition.

    Args:
        active_census: Current or forecasted count of active patients in the ED.
        recent_arrivals_60m: Rolling arrival count over past 60 minutes.
        high_acuity_ratio: Fraction of active patients with triage ESI <= 2.
        config: Optional configuration overriding weights and reference parameters.

    Returns:
        Structured dictionary containing score, state, component scores, weights,
        and human-readable decomposition.

    Raises:
        ValueError: If active_census or recent_arrivals_60m is negative, or
            high_acuity_ratio is outside [0.0, 1.0].
    """
    if active_census < 0:
        raise ValueError(f"active_census cannot be negative; got {active_census}")
    if recent_arrivals_60m < 0:
        raise ValueError(
            f"recent_arrivals_60m cannot be negative; got {recent_arrivals_60m}"
        )
    if not (0.0 <= high_acuity_ratio <= 1.0):
        raise ValueError(
            f"high_acuity_ratio must be in [0.0, 1.0]; got {high_acuity_ratio}"
        )

    cfg = config or DEFAULT_HEALTH_CONFIG

    # 1. Congestion Pressure: ratio of active census to capacity reference
    p_congestion = min(
        100.0, max(0.0, (float(active_census) / cfg.capacity_reference) * 100.0)
    )

    # 2. Arrival Pressure: ratio of arrival rate to reference intake
    p_arrivals = min(
        100.0,
        max(0.0, (float(recent_arrivals_60m) / cfg.arrival_rate_reference) * 100.0),
    )

    # 3. High-Acuity Pressure: ratio of high-acuity fraction to reference tolerance
    p_acuity = min(
        100.0, max(0.0, (float(high_acuity_ratio) / cfg.high_acuity_reference) * 100.0)
    )

    # Composite weighted score
    raw_score = (
        cfg.w_congestion * p_congestion
        + cfg.w_arrivals * p_arrivals
        + cfg.w_acuity * p_acuity
    )
    score = round(float(np.clip(raw_score, 0.0, 100.0)), 1)

    # Operational state mapping
    state: QueueHealthState = classify_queue_health_state(score, cfg.state_thresholds)

    # Determine dominant pressure vector
    weighted_contributions = {
        "congestion_pressure": cfg.w_congestion * p_congestion,
        "arrival_pressure": cfg.w_arrivals * p_arrivals,
        "high_acuity_pressure": cfg.w_acuity * p_acuity,
    }
    dominant_factor = max(
        weighted_contributions, key=lambda k: weighted_contributions[k]
    )

    factor_labels = {
        "congestion_pressure": "active bed occupancy",
        "arrival_pressure": "rapid presentation velocity",
        "high_acuity_pressure": "high-acuity clinical workload",
    }
    dominant_desc = factor_labels.get(dominant_factor, dominant_factor)

    summary = (
        f"Queue health is {state.value} (score: {score}/100). "
        f"Primary operational pressure driver is {dominant_desc}."
    )

    return {
        "score": score,
        "state": state.value,
        "components": {
            "congestion_pressure": round(p_congestion, 1),
            "arrival_pressure": round(p_arrivals, 1),
            "high_acuity_pressure": round(p_acuity, 1),
        },
        "weights": {
            "congestion": cfg.w_congestion,
            "arrivals": cfg.w_arrivals,
            "acuity": cfg.w_acuity,
        },
        "dominant_factor": dominant_factor,
        "summary": summary,
        "non_clinical_disclaimer": (
            "The Queue Health Score is an operational load metric synthesizing "
            "volume and flow velocity. It does not measure clinical severity "
            "or care quality."
        ),
    }
