"""Unit and integration tests for /predict/congestion endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from api.dependencies import get_congestion_predictor
from api.main import create_app
from queuemind.models.conformal import ConformalIntervalCalibrator
from queuemind.models.congestion import CongestionPredictor, RidgeCongestionModel


@pytest.fixture
def mock_congestion_predictor() -> CongestionPredictor:
    """Fixture providing an in-memory trained CongestionPredictor."""
    # Synthetic grid dataset
    data = {
        "current_active_census": [30, 35, 40, 45, 50],
        "recent_arrivals_15m": [2.0, 3.0, 4.0, 5.0, 3.0],
        "recent_arrivals_30m": [4.0, 6.0, 8.0, 10.0, 7.0],
        "recent_arrivals_60m": [8.0, 12.0, 15.0, 18.0, 14.0],
        "recent_arrivals_120m": [15.0, 20.0, 25.0, 30.0, 26.0],
        "recent_departures_15m": [2.0, 2.0, 3.0, 4.0, 3.0],
        "recent_departures_30m": [4.0, 5.0, 6.0, 8.0, 6.0],
        "recent_departures_60m": [8.0, 10.0, 12.0, 14.0, 12.0],
        "recent_departures_120m": [14.0, 18.0, 22.0, 26.0, 24.0],
        "net_flow_15m": [0.0, 1.0, 1.0, 1.0, 0.0],
        "net_flow_30m": [0.0, 1.0, 2.0, 2.0, 1.0],
        "net_flow_60m": [0.0, 2.0, 3.0, 4.0, 2.0],
        "flow_ratio_60m": [1.0, 1.2, 1.25, 1.28, 1.16],
        "high_acuity_census": [6.0, 8.0, 10.0, 12.0, 10.0],
        "high_acuity_ratio": [0.20, 0.23, 0.25, 0.27, 0.20],
        "hour_sin": [0.0, 0.5, 0.86, 1.0, 0.86],
        "hour_cos": [1.0, 0.86, 0.5, 0.0, -0.5],
        "dayofweek": [0, 1, 2, 3, 4],
        "is_weekend": [0, 0, 0, 0, 0],
    }
    targets = {
        "target_census_30m": [32.0, 37.0, 42.0, 47.0, 52.0],
        "target_census_60m": [34.0, 39.0, 44.0, 49.0, 54.0],
        "target_census_120m": [36.0, 41.0, 46.0, 51.0, 56.0],
    }
    X = pd.DataFrame(data)
    Y = pd.DataFrame(targets)

    model = RidgeCongestionModel(horizons=(30, 60, 120))
    model.fit(X, Y)

    # Setup calibrators
    preds = model.predict(X)
    calibrators = {}
    for h in (30, 60, 120):
        cal = ConformalIntervalCalibrator(default_coverage_level=0.90)
        cal.calibrate(
            y_true=Y[f"target_census_{h}m"], y_pred=preds[f"pred_census_{h}m"]
        )
        calibrators[h] = cal

    return CongestionPredictor(
        model=model,
        horizons=(30, 60, 120),
        feature_names=list(X.columns),
        calibrators=calibrators,
        model_name="ridge_congestion",
        model_version="0.1.0",
    )


class TestCongestionEndpoint:
    """Tests for /predict/congestion endpoint."""

    def test_model_unconfigured_returns_503(self) -> None:
        """Verify 503 when CONGESTION_MODEL_PATH is not configured."""
        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/predict/congestion",
            json={"current_active_census": 40},
        )
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]

    def test_successful_congestion_forecast(
        self, mock_congestion_predictor: CongestionPredictor
    ) -> None:
        """Verify successful multi-horizon forecasting with intervals and
        bottleneck signals."""
        app = create_app()
        app.dependency_overrides[get_congestion_predictor] = (
            lambda: mock_congestion_predictor
        )
        client = TestClient(app)

        payload = {
            "current_active_census": 42,
            "recent_arrivals_15m": 4.0,
            "recent_arrivals_30m": 7.0,
            "recent_arrivals_60m": 12.0,
            "recent_arrivals_120m": 22.0,
            "recent_departures_15m": 3.0,
            "recent_departures_30m": 5.0,
            "recent_departures_60m": 10.0,
            "recent_departures_120m": 20.0,
            "net_flow_15m": 1.0,
            "net_flow_30m": 2.0,
            "net_flow_60m": 2.0,
            "flow_ratio_60m": 1.2,
            "high_acuity_census": 9.0,
            "high_acuity_ratio": 0.22,
            "hour_sin": 0.5,
            "hour_cos": 0.86,
            "dayofweek": 2,
            "is_weekend": 0,
            "coverage_level": 0.90,
        }
        response = client.post("/predict/congestion", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["current_active_census"] == 42
        assert "forecasts" in data
        forecasts = data["forecasts"]
        assert "30m" in forecasts
        assert "60m" in forecasts
        assert "120m" in forecasts

        for h_key, h_val in [(30, "30m"), (60, "60m"), (120, "120m")]:
            fc = forecasts[h_val]
            assert fc["horizon_minutes"] == h_key
            assert fc["predicted_census"] >= 0.0
            assert fc["prediction_interval"] is not None

        assert "congestion_state" in data
        assert "bottleneck_indicators" in data
        assert data["model_name"] == "ridge_congestion"

    def test_negative_census_rejected_422(
        self, mock_congestion_predictor: CongestionPredictor
    ) -> None:
        """Verify 422 is returned when current_active_census is negative."""
        app = create_app()
        app.dependency_overrides[get_congestion_predictor] = (
            lambda: mock_congestion_predictor
        )
        client = TestClient(app)

        response = client.post(
            "/predict/congestion",
            json={"current_active_census": -5},
        )
        assert response.status_code == 422
