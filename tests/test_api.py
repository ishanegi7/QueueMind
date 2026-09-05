"""Unit and integration tests for FastAPI application core, health, and OpenAPI."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from api.config import Settings
from api.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a FastAPI TestClient."""
    app = create_app()
    return TestClient(app)


class TestApiCore:
    """Tests for API health, OpenAPI generation, CORS, and error handling."""

    def test_health_endpoint_healthy(self, client: TestClient) -> None:
        """Verify GET /health returns 200 with proper schema and service name."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "queuemind-api"
        assert data["version"] == "0.1.0"
        assert "models" in data
        assert "patient_flow" in data["models"]
        assert "congestion" in data["models"]

    def test_openapi_schema_generated(self, client: TestClient) -> None:
        """Verify OpenAPI JSON schema contains all defined routes."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "QueueMind API"
        paths = schema["paths"]
        assert "/health" in paths
        assert "/predict/patient-flow" in paths
        assert "/predict/congestion" in paths
        assert "/queue-health" in paths
        assert "/simulate/what-if" in paths

    def test_docs_and_redoc_endpoints(self, client: TestClient) -> None:
        """Verify Swagger UI and Redoc pages are accessible."""
        res_docs = client.get("/docs")
        assert res_docs.status_code == 200
        res_redoc = client.get("/redoc")
        assert res_redoc.status_code == 200

    def test_cors_headers_present(self, client: TestClient) -> None:
        """Verify CORS headers respond to configured origins."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert (
            response.headers.get("access-control-allow-origin")
            == "http://localhost:3000"
        )

    def test_not_found_endpoint(self, client: TestClient) -> None:
        """Verify 404 is returned cleanly for non-existent route."""
        response = client.get("/non-existent-endpoint")
        assert response.status_code == 404

    def test_custom_settings_factory(self) -> None:
        """Verify create_app respects custom Settings."""
        custom_settings = Settings(
            ENVIRONMENT="staging",
            ALLOWED_ORIGINS=["https://custom.hospital.org"],
        )
        custom_app = create_app(settings=custom_settings)
        test_client = TestClient(custom_app)
        res = test_client.get("/health")
        assert res.status_code == 200
        assert res.json()["environment"] == "staging"
