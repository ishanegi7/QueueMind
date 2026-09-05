"""Unit and integration tests for /simulate/what-if endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from api.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Fixture providing an API TestClient."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_trajectory_payload() -> dict:
    """Fixture providing baseline trajectory parameters."""
    start = pd.Timestamp("2026-03-01 10:00:00")
    time_steps = [(start + pd.Timedelta(minutes=15 * i)).isoformat() for i in range(5)]
    return {
        "time_steps": time_steps,
        "initial_census": 40.0,
        "arrivals": [5.0, 6.0, 4.0, 5.0],
        "departures": [4.0, 5.0, 5.0, 4.0],
        "high_acuity_ratio": 0.25,
    }


class TestSimulationEndpoint:
    """Tests for /simulate/what-if scenario execution, stability, and constraints."""

    def test_discharge_acceleration_simulation(
        self, client: TestClient, sample_trajectory_payload: dict
    ) -> None:
        """Verify +20% discharge acceleration scenario returns reduced census."""
        payload = {
            **sample_trajectory_payload,
            "scenario_type": "discharge_acceleration",
            "acceleration_rate": 0.20,
        }
        response = client.post("/simulate/what-if", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["scenario_type"] == "discharge_acceleration"
        assert data["final_simulated_census"] < data["final_baseline_census"]
        assert all(d <= 0.0 for d in data["census_delta"])
        assert data["stability"] in ["STABLE", "STRAINED", "UNSTABLE"]
        assert data["waiting_time_impact"]["status"] == "unavailable"
        assert len(data["limitations"]) > 0

    def test_capacity_reduction_simulation(
        self, client: TestClient, sample_trajectory_payload: dict
    ) -> None:
        """Verify capacity constraint scenario calculates peak bed overflow."""
        # Baseline peak is 42.0. Constraining capacity to 35 beds yields 7.0 overflow.
        payload = {
            **sample_trajectory_payload,
            "scenario_type": "capacity_reduction",
            "reduced_capacity": 35.0,
        }
        response = client.post("/simulate/what-if", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["scenario_type"] == "capacity_reduction"
        assert data["peak_delta"] == 7.0
        assert data["simulated_census"] == data["baseline_census"]
        assert (
            data["simulated_queue_health"]["score"]
            >= data["baseline_queue_health"]["score"]
        )

    def test_arrival_surge_simulation(
        self, client: TestClient, sample_trajectory_payload: dict
    ) -> None:
        """Verify arrival surge shock distributes presentations across surge
        intervals."""
        payload = {
            **sample_trajectory_payload,
            "scenario_type": "arrival_surge",
            "additional_arrivals": 10,
            "surge_duration_steps": 2,
            "surge_acuity_ratio": 0.40,
        }
        response = client.post("/simulate/what-if", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["scenario_type"] == "arrival_surge"
        assert data["simulated_arrivals"][0] == 5.0 + 5.0
        assert data["simulated_arrivals"][1] == 6.0 + 5.0
        assert data["peak_simulated_census"] > data["peak_baseline_census"]
        assert data["peak_delta"] > 0.0

    def test_missing_reduced_capacity_raises_422(
        self, client: TestClient, sample_trajectory_payload: dict
    ) -> None:
        """Verify 422 when capacity_reduction lacks reduced_capacity."""
        payload = {
            **sample_trajectory_payload,
            "scenario_type": "capacity_reduction",
        }
        response = client.post("/simulate/what-if", json=payload)
        assert response.status_code == 422

    def test_mismatched_arrivals_departures_raises_422(
        self, client: TestClient, sample_trajectory_payload: dict
    ) -> None:
        """Verify 422 when arrivals and departures array lengths differ."""
        payload = {
            **sample_trajectory_payload,
            "scenario_type": "discharge_acceleration",
            "departures": [4.0, 5.0],  # only 2 elements vs 4 arrivals
        }
        response = client.post("/simulate/what-if", json=payload)
        assert response.status_code == 422

    def test_negative_acceleration_rejected_422(
        self, client: TestClient, sample_trajectory_payload: dict
    ) -> None:
        """Verify 422 when acceleration_rate is negative."""
        payload = {
            **sample_trajectory_payload,
            "scenario_type": "discharge_acceleration",
            "acceleration_rate": -0.20,
        }
        response = client.post("/simulate/what-if", json=payload)
        assert response.status_code == 422
