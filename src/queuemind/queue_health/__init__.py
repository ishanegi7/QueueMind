"""Queue Health scoring package for QueueMind operational intelligence.

Provides:
- QueueHealthState: Operational categories (HEALTHY, MODERATE, BUSY, CRITICAL).
- QueueHealthStateThresholds: Configurable boundary definitions.
- QueueHealthConfig: Normalization references and component weighting.
- calculate_queue_health_score: Composite 0–100 pressure score generator.
"""

from queuemind.queue_health.score import (
    DEFAULT_HEALTH_CONFIG,
    QueueHealthConfig,
    calculate_queue_health_score,
)
from queuemind.queue_health.states import (
    DEFAULT_HEALTH_THRESHOLDS,
    QueueHealthState,
    QueueHealthStateThresholds,
    classify_queue_health_state,
)

__all__ = [
    "QueueHealthState",
    "QueueHealthStateThresholds",
    "DEFAULT_HEALTH_THRESHOLDS",
    "classify_queue_health_state",
    "QueueHealthConfig",
    "DEFAULT_HEALTH_CONFIG",
    "calculate_queue_health_score",
]
