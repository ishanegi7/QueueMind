"""Unit tests for QueueMind TreeSHAP explainability module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from queuemind.explainability import ShapExplainer
from queuemind.models.baseline import RidgeRegressionBaseline
from queuemind.models.predict import PatientFlowPredictor
from queuemind.models.train import (
    PROHIBITED_FEATURE_COLUMNS,
    XGBoostCandidate,
    chronological_split,
)


@pytest.fixture
def trained_xgb_and_data() -> tuple[XGBoostCandidate, pd.DataFrame, pd.DataFrame]:
    """Train a deterministic XGBoostCandidate on synthetic mock snapshots."""
    n = 60
    base_time = pd.Timestamp("2024-01-01 08:00:00")
    times = [base_time + pd.Timedelta(minutes=15 * i) for i in range(n)]

    rng = np.random.RandomState(42)
    acuities = rng.choice([1, 2, 3, 4, 5], size=n)
    ages = rng.randint(18, 90, size=n)
    heartrates = rng.uniform(60, 130, size=n)
    heartrates[::8] = np.nan
    genders = rng.choice(["M", "F"], size=n)
    transports = rng.choice(["AMBULANCE", "WALK IN"], size=n)
    active_census = rng.randint(5, 35, size=n)

    remaining = (
        (acuities * 35.0)
        + (ages * 0.4)
        + (active_census * 2.0)
        + rng.uniform(-10, 10, size=n)
    )
    remaining = np.maximum(10.0, remaining)

    df = pd.DataFrame(
        {
            "stay_id": [1000 + i for i in range(n)],
            "snapshot_time": times,
            "intime": [t - pd.Timedelta(minutes=30) for t in times],
            "outtime": [
                t + pd.Timedelta(minutes=float(rem)) for t, rem in zip(times, remaining)
            ],
            "acuity": acuities,
            "age": ages,
            "heartrate": heartrates,
            "gender": genders,
            "arrival_transport": transports,
            "active_census": active_census,
            "disposition": rng.choice(["HOME", "ADMITTED"], size=n),
            "remaining_time_minutes": remaining,
        }
    )

    train_df, _, test_df = chronological_split(
        df, time_col="snapshot_time", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    features = [
        "acuity",
        "age",
        "heartrate",
        "gender",
        "arrival_transport",
        "active_census",
    ]
    model = XGBoostCandidate(
        n_estimators=25,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        numeric_cols=["acuity", "age", "heartrate", "active_census"],
        categorical_cols=["gender", "arrival_transport"],
    )
    model.fit(train_df[features], train_df["remaining_time_minutes"])

    return model, train_df[features], test_df[features]


class TestShapExplainer:
    def test_initialization(
        self,
        trained_xgb_and_data: tuple[XGBoostCandidate, pd.DataFrame, pd.DataFrame],
    ) -> None:
        model, _, _ = trained_xgb_and_data
        explainer = ShapExplainer(model)

        assert explainer.base_value_ is not None
        assert isinstance(explainer.base_value_, float)
        assert len(explainer.orig_to_transformed_indices_) > 0

    def test_initialization_from_predictor(
        self,
        trained_xgb_and_data: tuple[XGBoostCandidate, pd.DataFrame, pd.DataFrame],
    ) -> None:
        model, _, _ = trained_xgb_and_data
        predictor = PatientFlowPredictor(
            model=model,
            model_name="xgb_flow",
            feature_names=list(model.feature_names_in_ or []),
        )
        explainer = ShapExplainer(predictor)
        assert explainer.base_value_ is not None

    def test_unfitted_model_raises(self) -> None:
        unfitted = XGBoostCandidate()
        with pytest.raises(NotFittedError, match="unfitted"):
            ShapExplainer(unfitted)

    def test_incompatible_model_raises(self) -> None:
        linear_model = RidgeRegressionBaseline()
        with pytest.raises(TypeError, match="requires XGBoostCandidate"):
            ShapExplainer(linear_model)

    def test_explain_single_structure_and_additivity(
        self,
        trained_xgb_and_data: tuple[XGBoostCandidate, pd.DataFrame, pd.DataFrame],
    ) -> None:
        model, _, test_df = trained_xgb_and_data
        explainer = ShapExplainer(model)

        sample = test_df.iloc[[0]]
        explanation = explainer.explain_single(sample)

        assert "prediction" in explanation
        assert "base_value" in explanation
        assert "features" in explanation

        features = explanation["features"]
        assert len(features) > 0

        total_shap = sum(f["shap_value"] for f in features)
        reconstructed_pred = explanation["base_value"] + total_shap

        # Additive property: base_value + sum(shap_values) ≈ model prediction
        assert pytest.approx(reconstructed_pred, abs=2.0) == explanation["prediction"]

        # Verify ranking order
        abs_shaps = [abs(f["shap_value"]) for f in features]
        assert abs_shaps == sorted(abs_shaps, reverse=True)
        assert [f["rank"] for f in features] == list(range(1, len(features) + 1))

    def test_explain_single_human_readable_mapping(
        self,
        trained_xgb_and_data: tuple[XGBoostCandidate, pd.DataFrame, pd.DataFrame],
    ) -> None:
        model, _, test_df = trained_xgb_and_data
        explainer = ShapExplainer(model)

        sample_dict = test_df.iloc[0].to_dict()
        explanation = explainer.explain_single(sample_dict)

        feature_names = [f["name"] for f in explanation["features"]]

        # Prohibit raw encoded dummy names
        for name in feature_names:
            assert not name.startswith("cat__")
            assert not name.startswith("num__")

        # Confirm original feature names are preserved
        assert "arrival_transport" in feature_names
        assert "gender" in feature_names
        assert "acuity" in feature_names
        assert "age" in feature_names

        # Direction checks
        for f in explanation["features"]:
            if f["shap_value"] > 0.001:
                assert f["direction"] == "increases_prediction"
            elif f["shap_value"] < -0.001:
                assert f["direction"] == "decreases_prediction"
            else:
                assert f["direction"] == "neutral"

    def test_explain_single_top_n(
        self,
        trained_xgb_and_data: tuple[XGBoostCandidate, pd.DataFrame, pd.DataFrame],
    ) -> None:
        model, _, test_df = trained_xgb_and_data
        explainer = ShapExplainer(model)

        explanation = explainer.explain_single(test_df.iloc[0].to_dict(), top_n=3)
        assert len(explanation["features"]) == 3

    def test_explain_single_rejects_leaked_columns(
        self,
        trained_xgb_and_data: tuple[XGBoostCandidate, pd.DataFrame, pd.DataFrame],
    ) -> None:
        model, _, test_df = trained_xgb_and_data
        explainer = ShapExplainer(model)

        leaked_dict = test_df.iloc[0].to_dict()
        for prohibited in PROHIBITED_FEATURE_COLUMNS:
            leaked_dict[prohibited] = 999.0
            with pytest.raises(ValueError, match="Data leakage detected"):
                explainer.explain_single(leaked_dict)
            del leaked_dict[prohibited]

    def test_explain_single_rejects_multi_row(
        self,
        trained_xgb_and_data: tuple[XGBoostCandidate, pd.DataFrame, pd.DataFrame],
    ) -> None:
        model, _, test_df = trained_xgb_and_data
        explainer = ShapExplainer(model)
        with pytest.raises(ValueError, match="expects 1 row"):
            explainer.explain_single(test_df.iloc[:2])

    def test_explain_global(
        self,
        trained_xgb_and_data: tuple[XGBoostCandidate, pd.DataFrame, pd.DataFrame],
    ) -> None:
        model, _, test_df = trained_xgb_and_data
        explainer = ShapExplainer(model)

        global_importance = explainer.explain_global(test_df, top_n=5)
        assert len(global_importance) <= 5

        # Check descending sort
        importances = [item["mean_abs_shap"] for item in global_importance]
        assert importances == sorted(importances, reverse=True)
        assert [item["rank"] for item in global_importance] == list(
            range(1, len(global_importance) + 1)
        )

        with pytest.raises(ValueError, match="empty DataFrame"):
            explainer.explain_global(pd.DataFrame())

    def test_deterministic_explanation(
        self,
        trained_xgb_and_data: tuple[XGBoostCandidate, pd.DataFrame, pd.DataFrame],
    ) -> None:
        model, _, test_df = trained_xgb_and_data
        explainer = ShapExplainer(model)

        sample = test_df.iloc[[0]]
        e1 = explainer.explain_single(sample)
        e2 = explainer.explain_single(sample)

        assert e1["prediction"] == e2["prediction"]
        assert e1["base_value"] == e2["base_value"]
        assert len(e1["features"]) == len(e2["features"])
        for f1, f2 in zip(e1["features"], e2["features"]):
            assert f1["name"] == f2["name"]
            assert f1["shap_value"] == f2["shap_value"]
