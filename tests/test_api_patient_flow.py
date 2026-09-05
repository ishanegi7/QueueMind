"""Unit and integration tests for /predict/patient-flow endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from api.dependencies import get_patient_flow_predictor
from api.main import create_app
from queuemind.explainability.shap_explainer import ShapExplainer
from queuemind.models.conformal import ConformalIntervalCalibrator
from queuemind.models.predict import PatientFlowPredictor
from queuemind.models.train import XGBoostCandidate


@pytest.fixture
def mock_predictor() -> PatientFlowPredictor:
    """Fixture creating an in-memory trained PatientFlowPredictor."""
    # Synthetic small training dataset
    data = {
        "acuity": [2, 3, 1, 4, 2],
        "temperature": [98.6, 99.1, 101.2, 97.8, 98.4],
        "heartrate": [75.0, 88.0, 110.0, 65.0, 82.0],
        "resprate": [16.0, 18.0, 24.0, 14.0, 16.0],
        "o2sat": [98.0, 97.0, 94.0, 99.0, 98.0],
        "sbp": [120.0, 130.0, 95.0, 140.0, 125.0],
        "dbp": [80.0, 85.0, 60.0, 90.0, 82.0],
        "pain": [5.0, 2.0, 8.0, 0.0, 4.0],
        "active_census": [35.0, 40.0, 45.0, 30.0, 38.0],
        "recent_arrivals_60m": [8.0, 10.0, 15.0, 6.0, 9.0],
        "remaining_time_minutes": [120.0, 180.0, 240.0, 90.0, 150.0],
    }
    df = pd.DataFrame(data)
    feature_names = [c for c in df.columns if c != "remaining_time_minutes"]

    model = XGBoostCandidate(n_estimators=10, max_depth=2, random_state=42)
    model.fit(df[feature_names], df["remaining_time_minutes"])

    calibrator = ConformalIntervalCalibrator(default_coverage_level=0.90)
    calibrator.calibrate(
        y_true=df["remaining_time_minutes"],
        y_pred=model.predict(df[feature_names]),
    )

    explainer = ShapExplainer(model)

    return PatientFlowPredictor(
        model=model,
        model_name="xgboost_candidate",
        model_version="0.1.0",
        feature_names=feature_names,
        calibrator=calibrator,
        explainer=explainer,
    )


class TestPatientFlowEndpoint:
    """Tests for /predict/patient-flow request handling, validations, and responses."""

    def test_model_unconfigured_returns_503(self) -> None:
        """Verify 503 is returned when PATIENT_FLOW_MODEL_PATH is unconfigured."""
        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/predict/patient-flow",
            json={"acuity": 3, "heartrate": 80.0, "active_census": 40.0},
        )
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]

    def test_successful_prediction_with_overridden_predictor(
        self, mock_predictor: PatientFlowPredictor
    ) -> None:
        """Verify successful prediction returns structured response."""
        app = create_app()
        app.dependency_overrides[get_patient_flow_predictor] = lambda: mock_predictor
        client = TestClient(app)

        payload = {
            "acuity": 2,
            "temperature": 98.6,
            "heartrate": 85.0,
            "resprate": 18.0,
            "o2sat": 97.0,
            "sbp": 125.0,
            "dbp": 80.0,
            "pain": 4.0,
            "active_census": 42.0,
            "recent_arrivals_60m": 10.0,
            "coverage_level": 0.90,
            "return_explanation": True,
        }
        response = client.post("/predict/patient-flow", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "predicted_remaining_time_minutes" in data
        assert data["predicted_remaining_time_minutes"] > 0.0
        assert data["unit"] == "minutes"
        assert data["model_name"] == "xgboost_candidate"
        assert data["model_version"] == "0.1.0"

        # Conformal interval
        interval = data["prediction_interval"]
        assert interval is not None
        assert interval["lower_minutes"] >= 0.0
        assert interval["upper_minutes"] >= interval["lower_minutes"]
        assert interval["coverage_level"] == 0.90
        assert interval["method"] == "split_conformal"

        # SHAP explanation
        explanation = data["explanation"]
        assert explanation is not None
        assert "features" in explanation
        assert len(explanation["features"]) > 0
        assert "interpretation" in explanation

    def test_rejection_of_prohibited_leakage_fields(
        self, mock_predictor: PatientFlowPredictor
    ) -> None:
        """Verify 422 is returned when leakage-prone fields are present."""
        app = create_app()
        app.dependency_overrides[get_patient_flow_predictor] = lambda: mock_predictor
        client = TestClient(app)

        prohibited_payloads = [
            {"acuity": 3, "stay_id": 12345},
            {"acuity": 3, "outtime": "2026-03-01 14:00:00"},
            {"acuity": 3, "disposition": "HOME"},
            {"acuity": 3, "remaining_time_minutes": 120.0},
        ]

        for payload in prohibited_payloads:
            response = client.post("/predict/patient-flow", json=payload)
            assert response.status_code == 422
            assert "Data leakage prohibited field" in str(response.json())

    def test_validation_bounds_errors(
        self, mock_predictor: PatientFlowPredictor
    ) -> None:
        """Verify 422 is returned when clinical parameters exceed valid
        physiological bounds."""
        app = create_app()
        app.dependency_overrides[get_patient_flow_predictor] = lambda: mock_predictor
        client = TestClient(app)

        # Invalid acuity > 5
        res_acuity = client.post("/predict/patient-flow", json={"acuity": 6})
        assert res_acuity.status_code == 422

        # Invalid temperature > 115
        res_temp = client.post("/predict/patient-flow", json={"temperature": 150.0})
        assert res_temp.status_code == 422

        # Invalid o2sat > 100
        res_o2 = client.post("/predict/patient-flow", json={"o2sat": 105.0})
        assert res_o2.status_code == 422
