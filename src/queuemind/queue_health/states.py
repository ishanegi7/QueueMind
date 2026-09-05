"""Operational Queue Health state classifications and configurable thresholds.

NON-CLINICAL DISCLAIMER:
These states represent mathematical queueing load and operational pressure tiers.
They do NOT represent clinically validated triage thresholds or medical severity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueueHealthState(str, Enum):
    """Operational state categories for Emergency Department queue pressure."""

    HEALTHY = "HEALTHY"
    MODERATE = "MODERATE"
    BUSY = "BUSY"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class QueueHealthStateThresholds:
    """Configurable boundaries for Queue Health State classification.

    Attributes:
        healthy_max: Upper boundary for HEALTHY state (default: 30.0).
        moderate_max: Upper boundary for MODERATE state (default: 60.0).
        busy_max: Upper boundary for BUSY state (default: 80.0).
    """

    healthy_max: float = 30.0
    moderate_max: float = 60.0
    busy_max: float = 80.0

    def __post_init__(self) -> None:
        if not (0.0 < self.healthy_max < self.moderate_max < self.busy_max < 100.0):
            raise ValueError(
                "Thresholds must satisfy 0 < healthy < moderate < busy < 100; "
                f"got healthy={self.healthy_max}, moderate={self.moderate_max}, "
                f"busy={self.busy_max}"
            )


DEFAULT_HEALTH_THRESHOLDS = QueueHealthStateThresholds()


def classify_queue_health_state(
    score: float,
    thresholds: QueueHealthStateThresholds | None = None,
) -> QueueHealthState:
    """Classify a numerical Queue Health Score into an operational state tier.

    Args:
        score: Computed Queue Health Score in [0.0, 100.0].
        thresholds: Optional custom threshold configuration.

    Returns:
        QueueHealthState enum value.

    Raises:
        ValueError: If score is outside [0.0, 100.0].
    """
    if not (0.0 <= score <= 100.0):
        raise ValueError(f"Queue Health Score must be in [0.0, 100.0]; got {score}")

    th = thresholds or DEFAULT_HEALTH_THRESHOLDS

    if score <= th.healthy_max:
        return QueueHealthState.HEALTHY
    elif score <= th.moderate_max:
        return QueueHealthState.MODERATE
    elif score <= th.busy_max:
        return QueueHealthState.BUSY
    else:
        return QueueHealthState.CRITICAL
