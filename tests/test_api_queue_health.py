"""Unit and integration tests for /queue-health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from api.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Fixture providing an API TestClient."""
    app = create_app()
    return TestClient(app)


class TestQueueHealthEndpoint:
    """Tests for /queue-health endpoint scoring, states, and validations."""

    def test_default_config_queue_health_healthy(self, client: TestClient) -> None:
        """Verify low-load input produces HEALTHY state."""
        payload = {
            "active_census": 10.0,
            "recent_arrivals_60m": 2.0,
            "high_acuity_ratio": 0.10,
        }
        response = client.post("/queue-health", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert 0.0 <= data["score"] <= 100.0
        assert data["state"] == "HEALTHY"
        assert "congestion_pressure" in data["components"]
        assert "arrival_pressure" in data["components"]
        assert "high_acuity_pressure" in data["components"]
        assert data["weights"]["congestion"] == 0.50
        assert data["weights"]["arrivals"] == 0.30
        assert data["weights"]["acuity"] == 0.20
        assert "summary" in data
        assert "non_clinical_disclaimer" in data

    def test_high_load_queue_health_busy_critical(self, client: TestClient) -> None:
        """Verify severe load produces BUSY or CRITICAL state."""
        payload = {
            "active_census": 65.0,  # exceeds nominal 50 beds
            "recent_arrivals_60m": 20.0,  # exceeds nominal 12/hr
            "high_acuity_ratio": 0.45,  # exceeds nominal 0.30
        }
        response = client.post("/queue-health", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["score"] >= 80.0
        assert data["state"] == "CRITICAL"
        assert data["dominant_factor"] in [
            "congestion_pressure",
            "arrival_pressure",
            "high_acuity_pressure",
        ]

    def test_custom_weights_and_references(self, client: TestClient) -> None:
        """Verify custom weights and references are accepted and applied."""
        payload = {
            "active_census": 30.0,
            "recent_arrivals_60m": 6.0,
            "high_acuity_ratio": 0.20,
            "w_congestion": 0.70,
            "w_arrivals": 0.20,
            "w_acuity": 0.10,
            "capacity_reference": 60.0,
            "arrival_rate_reference": 15.0,
        }
        response = client.post("/queue-health", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["weights"]["congestion"] == 0.70
        assert data["weights"]["arrivals"] == 0.20
        assert data["weights"]["acuity"] == 0.10

    def test_invalid_weights_sum_returns_400(self, client: TestClient) -> None:
        """Verify 400 when custom weights do not sum to 1.0."""
        payload = {
            "active_census": 30.0,
            "recent_arrivals_60m": 6.0,
            "high_acuity_ratio": 0.20,
            "w_congestion": 0.50,
            "w_arrivals": 0.50,
            "w_acuity": 0.50,  # sum = 1.5
        }
        response = client.post("/queue-health", json=payload)
        assert response.status_code == 400
        assert "Invalid queue health configuration" in response.json()["detail"]

    def test_negative_active_census_rejected_422(self, client: TestClient) -> None:
        """Verify 422 when active_census is negative."""
        payload = {
            "active_census": -10.0,
            "recent_arrivals_60m": 5.0,
            "high_acuity_ratio": 0.20,
        }
        response = client.post("/queue-health", json=payload)
        assert response.status_code == 422

    def test_acuity_ratio_out_of_range_rejected_422(self, client: TestClient) -> None:
        """Verify 422 when high_acuity_ratio is > 1.0."""
        payload = {
            "active_census": 30.0,
            "recent_arrivals_60m": 5.0,
            "high_acuity_ratio": 1.5,
        }
        response = client.post("/queue-health", json=payload)
        assert response.status_code == 422
