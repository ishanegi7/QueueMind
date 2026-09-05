"""
Feature engineering module for QueueMind.

Provides point-in-time leakage-safe extraction of patient, temporal,
and congestion features, alongside snapshot dataset generators.
"""

from queuemind.features.bottleneck_features import (
    DEFAULT_BOTTLENECK_THRESHOLDS,
    DEFAULT_CAPACITY_THRESHOLDS,
    CongestionLevel,
    classify_congestion_state,
    detect_bottleneck_indicators,
)
from queuemind.features.congestion_features import calculate_congestion_features
from queuemind.features.patient_features import extract_patient_features
from queuemind.features.snapshots import (
    create_patient_snapshot,
    create_remaining_time_label,
    generate_snapshots_dataset,
)
from queuemind.features.temporal_features import extract_temporal_features
from queuemind.features.time_grid import (
    create_congestion_targets,
    extract_department_snapshot_features,
    generate_congestion_dataset,
    generate_time_grid,
)

__all__ = [
    "extract_patient_features",
    "extract_temporal_features",
    "calculate_congestion_features",
    "create_patient_snapshot",
    "create_remaining_time_label",
    "generate_snapshots_dataset",
    "generate_time_grid",
    "extract_department_snapshot_features",
    "create_congestion_targets",
    "generate_congestion_dataset",
    "CongestionLevel",
    "DEFAULT_CAPACITY_THRESHOLDS",
    "DEFAULT_BOTTLENECK_THRESHOLDS",
    "classify_congestion_state",
    "detect_bottleneck_indicators",
]
