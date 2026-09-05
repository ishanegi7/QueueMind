"""Unit tests for Queue Health scoring, states, and component decomposition."""

from __future__ import annotations

import pytest

from queuemind.queue_health.score import (
    QueueHealthConfig,
    calculate_queue_health_score,
)
from queuemind.queue_health.states import (
    QueueHealthState,
    QueueHealthStateThresholds,
    classify_queue_health_state,
)


class TestQueueHealthStates:
    """Test suite for operational queue health state classification."""

    def test_default_threshold_boundaries(self) -> None:
        assert classify_queue_health_state(0.0) == QueueHealthState.HEALTHY
        assert classify_queue_health_state(30.0) == QueueHealthState.HEALTHY
        assert classify_queue_health_state(30.1) == QueueHealthState.MODERATE
        assert classify_queue_health_state(60.0) == QueueHealthState.MODERATE
        assert classify_queue_health_state(60.1) == QueueHealthState.BUSY
        assert classify_queue_health_state(80.0) == QueueHealthState.BUSY
        assert classify_queue_health_state(80.1) == QueueHealthState.CRITICAL
        assert classify_queue_health_state(100.0) == QueueHealthState.CRITICAL

    def test_custom_thresholds(self) -> None:
        custom_th = QueueHealthStateThresholds(
            healthy_max=20.0,
            moderate_max=40.0,
            busy_max=70.0,
        )
        assert classify_queue_health_state(15.0, custom_th) == QueueHealthState.HEALTHY
        assert classify_queue_health_state(25.0, custom_th) == QueueHealthState.MODERATE
        assert classify_queue_health_state(55.0, custom_th) == QueueHealthState.BUSY
        assert classify_queue_health_state(75.0, custom_th) == QueueHealthState.CRITICAL

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError, match="Thresholds must satisfy"):
            QueueHealthStateThresholds(healthy_max=60.0, moderate_max=30.0)

    def test_score_outside_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            classify_queue_health_state(-1.0)
        with pytest.raises(ValueError, match="must be in"):
            classify_queue_health_state(100.5)


class TestQueueHealthScore:
    """Test suite for Queue Health Score computation and decomposition."""

    def test_zero_inputs_healthy(self) -> None:
        res = calculate_queue_health_score(
            active_census=0,
            recent_arrivals_60m=0,
            high_acuity_ratio=0.0,
        )
        assert res["score"] == 0.0
        assert res["state"] == QueueHealthState.HEALTHY.value
        assert res["components"]["congestion_pressure"] == 0.0
        assert res["components"]["arrival_pressure"] == 0.0
        assert res["components"]["high_acuity_pressure"] == 0.0

    def test_score_range_and_monotonicity(self) -> None:
        low = calculate_queue_health_score(15, 3, 0.10)
        med = calculate_queue_health_score(35, 7, 0.25)
        high = calculate_queue_health_score(55, 12, 0.45)

        assert 0.0 <= low["score"] < med["score"] < high["score"] <= 100.0
        assert low["state"] in [
            QueueHealthState.HEALTHY.value,
            QueueHealthState.MODERATE.value,
        ]
        assert high["state"] in [
            QueueHealthState.BUSY.value,
            QueueHealthState.CRITICAL.value,
        ]

    def test_extreme_clipping_at_100(self) -> None:
        res = calculate_queue_health_score(
            active_census=250,
            recent_arrivals_60m=80,
            high_acuity_ratio=1.0,
        )
        assert res["score"] == 100.0
        assert res["state"] == QueueHealthState.CRITICAL.value
        assert res["components"]["congestion_pressure"] == 100.0

    def test_dominant_factor_identification(self) -> None:
        # High census alone
        res_cong = calculate_queue_health_score(60, 0, 0.0)
        assert res_cong["dominant_factor"] == "congestion_pressure"
        assert "active bed occupancy" in res_cong["summary"]

        # High arrivals alone
        res_arr = calculate_queue_health_score(0, 15, 0.0)
        assert res_arr["dominant_factor"] == "arrival_pressure"
        assert "rapid presentation velocity" in res_arr["summary"]

        # High acuity alone
        res_acu = calculate_queue_health_score(0, 0, 0.80)
        assert res_acu["dominant_factor"] == "high_acuity_pressure"
        assert "high-acuity clinical workload" in res_acu["summary"]

    def test_custom_weights(self) -> None:
        cfg = QueueHealthConfig(
            w_congestion=0.70,
            w_arrivals=0.20,
            w_acuity=0.10,
        )
        res = calculate_queue_health_score(
            active_census=40,
            recent_arrivals_60m=5,
            high_acuity_ratio=0.20,
            config=cfg,
        )
        assert res["weights"]["congestion"] == 0.70
        assert res["weights"]["arrivals"] == 0.20
        assert res["weights"]["acuity"] == 0.10

    def test_invalid_config_raises(self) -> None:
        with pytest.raises(ValueError, match="Component weights must sum to 1.0"):
            QueueHealthConfig(w_congestion=0.6, w_arrivals=0.6, w_acuity=0.1)

        with pytest.raises(ValueError, match="strictly positive"):
            QueueHealthConfig(w_congestion=0.8, w_arrivals=0.2, w_acuity=0.0)

        with pytest.raises(ValueError, match="capacity_reference must be positive"):
            QueueHealthConfig(capacity_reference=0.0)

        with pytest.raises(ValueError, match="arrival_rate_reference must be positive"):
            QueueHealthConfig(arrival_rate_reference=-5.0)

        with pytest.raises(ValueError, match="high_acuity_reference must be in"):
            QueueHealthConfig(high_acuity_reference=1.5)

    def test_negative_input_validation(self) -> None:
        with pytest.raises(ValueError, match="active_census cannot be negative"):
            calculate_queue_health_score(-5, 5, 0.2)

        with pytest.raises(ValueError, match="recent_arrivals_60m cannot be negative"):
            calculate_queue_health_score(20, -1, 0.2)

        with pytest.raises(ValueError, match="high_acuity_ratio must be in"):
            calculate_queue_health_score(20, 5, 1.2)
