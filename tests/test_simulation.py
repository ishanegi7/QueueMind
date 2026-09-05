"""Unit tests for the QueueMind what-if simulation and counterfactual engine."""

from __future__ import annotations

import pandas as pd
import pytest

from queuemind.simulation.what_if import (
    WAITING_TIME_UNAVAILABLE_PAYLOAD,
    BaselineTrajectory,
    QueueStability,
    evaluate_queue_stability,
    simulate_arrival_surge,
    simulate_capacity_reduction,
    simulate_discharge_acceleration,
)


@pytest.fixture
def sample_trajectory() -> BaselineTrajectory:
    """Fixture providing a standard 4-step (1-hour) baseline trajectory."""
    start = pd.Timestamp("2026-03-01 10:00:00")
    time_steps = [start + pd.Timedelta(minutes=15 * i) for i in range(5)]
    return BaselineTrajectory(
        time_steps=time_steps,
        initial_census=40.0,
        arrivals=[5.0, 6.0, 4.0, 5.0],
        departures=[4.0, 5.0, 5.0, 4.0],
        high_acuity_ratio=0.25,
    )


class TestBaselineTrajectory:
    """Tests for BaselineTrajectory initialization, validation, and dynamics."""

    def test_census_trajectory_calculation(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify deterministic flow conservation C(t+1) = max(0, C(t) + Arr - Dep)."""
        # t0: 40.0
        # t1: 40 + 5 - 4 = 41.0
        # t2: 41 + 6 - 5 = 42.0
        # t3: 42 + 4 - 5 = 41.0
        # t4: 41 + 5 - 4 = 42.0
        expected = [40.0, 41.0, 42.0, 41.0, 42.0]
        assert sample_trajectory.census_trajectory == expected

    def test_census_clamped_at_zero(self) -> None:
        """Verify active census clamped at zero when departures exceed volume."""
        start = pd.Timestamp("2026-03-01 10:00:00")
        traj = BaselineTrajectory(
            time_steps=[start, start + pd.Timedelta(minutes=15)],
            initial_census=2.0,
            arrivals=[1.0],
            departures=[10.0],  # departures exceed 2 + 1 = 3
            high_acuity_ratio=0.1,
        )
        assert traj.census_trajectory == [2.0, 0.0]

    def test_mismatched_arrivals_departures_raises(self) -> None:
        """Verify error when arrivals and departures lengths differ."""
        start = pd.Timestamp("2026-03-01 10:00:00")
        with pytest.raises(ValueError, match="identical lengths"):
            BaselineTrajectory(
                time_steps=[start, start + pd.Timedelta(minutes=15)],
                initial_census=10.0,
                arrivals=[5.0, 3.0],
                departures=[4.0],
            )

    def test_mismatched_time_steps_raises(self) -> None:
        """Verify error when time_steps length does not equal len(arrivals) + 1."""
        start = pd.Timestamp("2026-03-01 10:00:00")
        with pytest.raises(ValueError, match="time_steps must have length"):
            BaselineTrajectory(
                time_steps=[start, start + pd.Timedelta(minutes=15)],
                initial_census=10.0,
                arrivals=[5.0, 3.0],
                departures=[4.0, 3.0],
            )

    def test_negative_initial_census_raises(self) -> None:
        """Verify error when initial_census is negative."""
        start = pd.Timestamp("2026-03-01 10:00:00")
        with pytest.raises(ValueError, match="initial_census cannot be negative"):
            BaselineTrajectory(
                time_steps=[start, start + pd.Timedelta(minutes=15)],
                initial_census=-5.0,
                arrivals=[2.0],
                departures=[2.0],
            )

    def test_invalid_high_acuity_ratio_raises(self) -> None:
        """Verify error when high_acuity_ratio is outside [0.0, 1.0]."""
        start = pd.Timestamp("2026-03-01 10:00:00")
        with pytest.raises(ValueError, match="high_acuity_ratio must be in"):
            BaselineTrajectory(
                time_steps=[start, start + pd.Timedelta(minutes=15)],
                initial_census=10.0,
                arrivals=[2.0],
                departures=[2.0],
                high_acuity_ratio=1.5,
            )


class TestDischargeAccelerationSimulation:
    """Tests for Scenario A: Discharge throughput acceleration."""

    def test_accelerated_departures_reduce_census(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify +20% acceleration increases departures and reduces end census."""
        result = simulate_discharge_acceleration(
            sample_trajectory, acceleration_rate=0.20
        )

        # Baseline departures: [4.0, 5.0, 5.0, 4.0]
        # +20%: [4.8, 6.0, 6.0, 4.8]
        assert result.simulated_departures == [4.8, 6.0, 6.0, 4.8]
        assert result.final_simulated_census < result.final_baseline_census
        assert result.peak_simulated_census <= result.peak_baseline_census
        assert all(delta <= 0.0 for delta in result.census_delta)
        assert result.scenario_type == "discharge_acceleration"

    def test_zero_acceleration_leaves_census_identical(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify 0% acceleration produces exact match to baseline."""
        result = simulate_discharge_acceleration(
            sample_trajectory, acceleration_rate=0.0
        )
        assert result.simulated_census == sample_trajectory.census_trajectory
        assert result.peak_delta == 0.0
        assert all(delta == 0.0 for delta in result.census_delta)

    def test_negative_acceleration_raises(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify negative acceleration rate is rejected."""
        with pytest.raises(ValueError, match="acceleration_rate cannot be negative"):
            simulate_discharge_acceleration(sample_trajectory, acceleration_rate=-0.10)

    def test_baseline_not_mutated(self, sample_trajectory: BaselineTrajectory) -> None:
        """Verify original baseline trajectory lists remain unmodified."""
        orig_arrivals = list(sample_trajectory.arrivals)
        orig_departures = list(sample_trajectory.departures)
        simulate_discharge_acceleration(sample_trajectory, acceleration_rate=0.50)
        assert sample_trajectory.arrivals == orig_arrivals
        assert sample_trajectory.departures == orig_departures

    def test_non_causal_payload_included(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify responsible non-causal disclaimer payload is attached."""
        result = simulate_discharge_acceleration(
            sample_trajectory, acceleration_rate=0.20
        )
        assert result.waiting_time_impact["status"] == "unavailable"
        assert "scientifically unsupported" in result.waiting_time_impact["reason"]


class TestCapacityReductionSimulation:
    """Tests for Scenario B: Capacity constraint reduction."""

    def test_capacity_reduction_overflow_and_health(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify constrained capacity flags overflow and elevates health score."""
        # Baseline peak is 42.0. Constraining capacity to 35 yields overflow of 7.0.
        result = simulate_capacity_reduction(sample_trajectory, reduced_capacity=35.0)

        assert result.scenario_type == "capacity_reduction"
        assert result.peak_delta == 7.0  # peak overflow
        assert result.simulated_census == result.baseline_census
        # Strained or Unstable queue health because capacity is reduced
        assert result.simulated_queue_health["score"] >= (
            result.baseline_queue_health["score"]
        )

    def test_non_positive_capacity_raises(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify zero or negative capacity raises ValueError."""
        with pytest.raises(ValueError, match="reduced_capacity must be positive"):
            simulate_capacity_reduction(sample_trajectory, reduced_capacity=0.0)
        with pytest.raises(ValueError, match="reduced_capacity must be positive"):
            simulate_capacity_reduction(sample_trajectory, reduced_capacity=-10.0)


class TestArrivalSurgeSimulation:
    """Tests for Scenario C: Arrival surge shocks."""

    def test_arrival_surge_distribution(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify additional arrivals are uniformly spread across surge steps."""
        # +10 arrivals over 2 steps -> +5 per step
        result = simulate_arrival_surge(
            sample_trajectory,
            additional_arrivals=10,
            surge_duration_steps=2,
            surge_acuity_ratio=0.40,
        )

        assert result.scenario_type == "arrival_surge"
        assert result.simulated_arrivals[0] == sample_trajectory.arrivals[0] + 5.0
        assert result.simulated_arrivals[1] == sample_trajectory.arrivals[1] + 5.0
        assert result.simulated_arrivals[2] == sample_trajectory.arrivals[2]
        assert result.simulated_arrivals[3] == sample_trajectory.arrivals[3]
        assert result.peak_simulated_census > result.peak_baseline_census
        assert result.simulated_census[-1] > result.baseline_census[-1]

    def test_negative_additional_arrivals_raises(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify negative arrival surge raises ValueError."""
        with pytest.raises(ValueError, match="additional_arrivals cannot be negative"):
            simulate_arrival_surge(sample_trajectory, additional_arrivals=-5)

    def test_zero_surge_duration_raises(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify surge duration < 1 raises ValueError."""
        with pytest.raises(ValueError, match="surge_duration_steps must be at least 1"):
            simulate_arrival_surge(sample_trajectory, surge_duration_steps=0)

    def test_surge_duration_exceeding_trajectory_clamped(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify surge duration exceeding trajectory length is safely clamped."""
        # 4 intervals total, duration requested = 10
        result = simulate_arrival_surge(
            sample_trajectory,
            additional_arrivals=8,
            surge_duration_steps=10,
        )
        # Should be distributed over all 4 intervals: 8 / 4 = 2.0 per step
        for sim, base in zip(result.simulated_arrivals, sample_trajectory.arrivals):
            assert sim == pytest.approx(base + 2.0)


class TestQueueStabilityEvaluation:
    """Tests for operational queue stability classifier."""

    def test_stable_condition(self) -> None:
        """Verify STABLE when departures >= arrivals and census does not grow."""
        status = evaluate_queue_stability(
            arrivals=[4.0, 4.0],
            departures=[5.0, 5.0],
            initial_census=30.0,
            final_census=28.0,
        )
        assert status == QueueStability.STABLE

    def test_strained_condition(self) -> None:
        """Verify STRAINED when arrivals exceed departures slightly but contained."""
        status = evaluate_queue_stability(
            arrivals=[5.0, 5.0],
            departures=[4.0, 4.0],
            initial_census=30.0,
            final_census=32.0,
        )
        assert status == QueueStability.STRAINED

    def test_unstable_on_large_net_flow(self) -> None:
        """Verify UNSTABLE when net flow exceeds 10.0."""
        status = evaluate_queue_stability(
            arrivals=[15.0, 15.0],
            departures=[5.0, 5.0],
            initial_census=30.0,
            final_census=50.0,
        )
        assert status == QueueStability.UNSTABLE

    def test_unstable_on_runaway_census(self) -> None:
        """Verify UNSTABLE when final census exceeds 1.5x initial census."""
        status = evaluate_queue_stability(
            arrivals=[6.0, 6.0],
            departures=[2.0, 2.0],
            initial_census=10.0,
            final_census=18.0,  # 18 > 1.5 * 10
        )
        assert status == QueueStability.UNSTABLE


class TestScenarioResultSerialization:
    """Tests for ScenarioResult dictionary serialization and determinism."""

    def test_serialization_to_dict(self, sample_trajectory: BaselineTrajectory) -> None:
        """Verify to_dict returns JSON-serializable structure with ISO timestamps."""
        result = simulate_discharge_acceleration(
            sample_trajectory, acceleration_rate=0.25
        )
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["scenario_name"] == "+25% Discharge Acceleration"
        assert d["scenario_type"] == "discharge_acceleration"
        assert isinstance(d["time_steps"], list)
        assert isinstance(d["time_steps"][0], str)
        assert "T" in d["time_steps"][0]  # ISO format
        assert d["waiting_time_impact"] == WAITING_TIME_UNAVAILABLE_PAYLOAD
        assert len(d["limitations"]) > 0

    def test_simulation_determinism(
        self, sample_trajectory: BaselineTrajectory
    ) -> None:
        """Verify identical inputs produce deterministic identical outputs."""
        res1 = simulate_discharge_acceleration(
            sample_trajectory, acceleration_rate=0.20
        )
        res2 = simulate_discharge_acceleration(
            sample_trajectory, acceleration_rate=0.20
        )

        assert res1.simulated_census == res2.simulated_census
        assert res1.peak_simulated_census == res2.peak_simulated_census
        assert res1.stability == res2.stability
