"""Operational flow indicators and department congestion state classification.

Provides:
1. Operational Bottleneck Indicators: Mathematical queue dynamics signals
   (rising census velocity, arrival pressure, low departure throughput,
   sustained net accumulation, high-acuity concentration).
2. Congestion State Classification: Operational status categories
   (HEALTHY, MODERATE, BUSY, CRITICAL) using configurable capacity thresholds.

NON-CLINICAL DISCLAIMER:
These indicators quantify queueing network state and flow velocities.
They do NOT diagnose clinical etiology, provider competence, or medical triage quality.
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class CongestionLevel(str, Enum):
    """Operational congestion categories for ED situational awareness."""

    HEALTHY = "HEALTHY"
    MODERATE = "MODERATE"
    BUSY = "BUSY"
    CRITICAL = "CRITICAL"


DEFAULT_CAPACITY_THRESHOLDS: dict[str, int] = {
    "healthy_max": 25,
    "moderate_max": 45,
    "busy_max": 65,
}

DEFAULT_BOTTLENECK_THRESHOLDS: dict[str, float] = {
    "net_flow_30m_velocity": 3.0,
    "net_flow_60m_velocity": 5.0,
    "high_arrival_rate_60m": 8.0,
    "low_departure_rate_60m": 2.0,
    "low_departure_min_census": 20.0,
    "high_acuity_ratio_threshold": 0.40,
}


def classify_congestion_state(
    active_census: int,
    capacity_thresholds: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Classify current operational state into situational congestion categories.

    Args:
        active_census: Real-time count of patients active in the ED.
        capacity_thresholds: Optional dictionary mapping 'healthy_max',
            'moderate_max', and 'busy_max'. If None, uses default prototype thresholds.

    Returns:
        Dictionary containing:
        - 'state': String representation of CongestionLevel.
        - 'active_census': Current patient count.
        - 'description': Human-readable operational context.

    Raises:
        ValueError: If active_census is negative.
    """
    if active_census < 0:
        raise ValueError(f"active_census cannot be negative; got {active_census}")

    thresholds = capacity_thresholds or DEFAULT_CAPACITY_THRESHOLDS
    h_max = thresholds.get("healthy_max", 25)
    m_max = thresholds.get("moderate_max", 45)
    b_max = thresholds.get("busy_max", 65)

    if active_census <= h_max:
        level = CongestionLevel.HEALTHY
        desc = (
            f"Census ({active_census}) is within baseline capacity (<= {h_max}); "
            "flow throughput is stable."
        )
    elif active_census <= m_max:
        level = CongestionLevel.MODERATE
        desc = (
            f"Census ({active_census}) is elevated ({h_max + 1}–{m_max}); "
            "moderate queue accumulation observed."
        )
    elif active_census <= b_max:
        level = CongestionLevel.BUSY
        desc = (
            f"Census ({active_census}) is high ({m_max + 1}–{b_max}); "
            "approaching department surge capacity."
        )
    else:
        level = CongestionLevel.CRITICAL
        desc = (
            f"Census ({active_census}) exceeds surge boundary (> {b_max}); "
            "critical operational congestion detected."
        )

    return {
        "state": level.value,
        "active_census": active_census,
        "description": desc,
    }


def detect_bottleneck_indicators(
    snapshot_features: dict[str, Any] | pd.Series,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate operational bottleneck indicators from department snapshot features.

    Evaluates five mathematical queue dynamics indicators:
    1. rising_census_velocity: Rapid net intake accumulation over 30m/60m.
    2. high_arrival_pressure: Rolling arrivals substantially above baseline.
    3. low_departure_throughput: Depressed discharge velocity during high census.
    4. sustained_positive_net_flow: Consistent positive net flow across 15m, 30m, 60m.
    5. acuity_concentration: Disproportionate ratio of high-acuity (ESI 1–2) patients.

    Args:
        snapshot_features: Dictionary or Series of snapshot features.
        thresholds: Optional custom threshold overrides.

    Returns:
        Structured dictionary with indicator flags, active count, and summary.
    """
    feats = (
        snapshot_features.to_dict()
        if isinstance(snapshot_features, pd.Series)
        else dict(snapshot_features)
    )
    th = thresholds or DEFAULT_BOTTLENECK_THRESHOLDS

    census = int(feats.get("current_active_census", 0))
    net_15m = int(feats.get("net_flow_15m", 0))
    net_30m = int(feats.get("net_flow_30m", 0))
    net_60m = int(feats.get("net_flow_60m", 0))
    arr_60m = float(feats.get("recent_arrivals_60m", 0))
    dep_60m = float(feats.get("recent_departures_60m", 0))
    high_acuity_ratio = float(feats.get("high_acuity_ratio", 0.0))

    # 1. Rising census velocity
    rising_velocity = bool(
        net_30m >= th.get("net_flow_30m_velocity", 3.0)
        or net_60m >= th.get("net_flow_60m_velocity", 5.0)
    )

    # 2. High arrival pressure
    high_arrival_pressure = bool(arr_60m >= th.get("high_arrival_rate_60m", 8.0))

    # 3. Low departure throughput (active bottleneck: high demand but stalled discharge)
    low_dep_thresh = th.get("low_departure_rate_60m", 2.0)
    low_dep_census = th.get("low_departure_min_census", 20.0)
    low_departure_throughput = bool(
        dep_60m <= low_dep_thresh and census >= low_dep_census
    )

    # 4. Sustained positive net flow
    sustained_net_flow = bool(net_15m > 0 and net_30m > 0 and net_60m > 0)

    # 5. Acuity concentration
    acuity_thresh = th.get("high_acuity_ratio_threshold", 0.40)
    acuity_concentration = bool(high_acuity_ratio >= acuity_thresh and census >= 5)

    indicators: dict[str, bool] = {
        "rising_census_velocity": rising_velocity,
        "high_arrival_pressure": high_arrival_pressure,
        "low_departure_throughput": low_departure_throughput,
        "sustained_positive_net_flow": sustained_net_flow,
        "acuity_concentration": acuity_concentration,
    }

    active_indicators = [name for name, active in indicators.items() if active]
    severity_score = len(active_indicators)

    messages: list[str] = []
    if rising_velocity:
        messages.append("Accelerating patient accumulation in recent rolling windows.")
    if high_arrival_pressure:
        messages.append("Elevated presentation rate exerting intake pressure.")
    if low_departure_throughput:
        messages.append("Depressed departure velocity despite elevated active census.")
    if sustained_net_flow:
        messages.append(
            "Continuous positive net flow across 15m, 30m, and 60m horizons."
        )
    if acuity_concentration:
        messages.append(
            f"High-acuity encounters represent {round(high_acuity_ratio * 100, 1)}% "
            "of active bed census."
        )

    summary = (
        " ".join(messages)
        if messages
        else "Flow indicators within standard operational variance."
    )

    return {
        "indicators": indicators,
        "active_indicators": active_indicators,
        "severity_score": severity_score,
        "summary": summary,
        "non_clinical_disclaimer": (
            "Operational flow indicators describe mathematical queue dynamics and "
            "velocities, not clinical etiology or provider performance."
        ),
    }
