"""Unit tests for QueueMind models, chronological splitting, baselines,
and evaluation.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from queuemind.models.baseline import (
    AcuityStratifiedMedianBaseline,
    GlobalMedianBaseline,
    RidgeRegressionBaseline,
)
from queuemind.models.evaluate import (
    evaluate_regression,
    evaluate_subgroups,
    format_metrics_summary,
)
from queuemind.models.predict import (
    PatientFlowPredictor,
    load_predictor,
    save_predictor,
)
from queuemind.models.train import (
    PROHIBITED_FEATURE_COLUMNS,
    XGBoostCandidate,
    build_preprocessor,
    chronological_split,
    get_model_feature_names,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_snapshot_df() -> pd.DataFrame:
    """Deterministic snapshot dataset for testing splits and models."""
    n = 60
    base_time = pd.Timestamp("2024-01-01 08:00:00")
    times = [base_time + pd.Timedelta(minutes=15 * i) for i in range(n)]

    rng = np.random.RandomState(42)
    acuities = rng.choice([1, 2, 3, 4, 5], size=n)
    ages = rng.randint(18, 90, size=n)
    heartrates = rng.uniform(60, 130, size=n)
    heartrates[::7] = np.nan  # Inject periodic NaNs
    genders = rng.choice(["M", "F"], size=n)
    transports = rng.choice(["AMBULANCE", "WALK IN"], size=n)

    # Remaining time loosely correlates with acuity and age
    remaining_minutes = (acuities * 40.0) + (ages * 0.5) + rng.uniform(-15, 15, size=n)
    remaining_minutes = np.maximum(10.0, remaining_minutes)

    df = pd.DataFrame(
        {
            "stay_id": [1000 + i for i in range(n)],
            "snapshot_time": times,
            "intime": [t - pd.Timedelta(minutes=30) for t in times],
            "outtime": [
                t + pd.Timedelta(minutes=float(rem))
                for t, rem in zip(times, remaining_minutes)
            ],
            "acuity": acuities,
            "age": ages,
            "heartrate": heartrates,
            "gender": genders,
            "arrival_transport": transports,
            "disposition": rng.choice(["HOME", "ADMITTED"], size=n),
            "remaining_time_minutes": remaining_minutes,
        }
    )
    return df


# ---------------------------------------------------------------------------
# 1. Chronological Split Tests
# ---------------------------------------------------------------------------


class TestChronologicalSplit:
    def test_chronological_ordering(self, mock_snapshot_df: pd.DataFrame) -> None:
        train, val, test = chronological_split(
            mock_snapshot_df,
            time_col="snapshot_time",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(mock_snapshot_df)

        # Strict temporal non-overlapping boundary
        assert train["snapshot_time"].max() <= val["snapshot_time"].min()
        assert val["snapshot_time"].max() <= test["snapshot_time"].min()

    def test_deterministic_split(self, mock_snapshot_df: pd.DataFrame) -> None:
        t1, v1, s1 = chronological_split(mock_snapshot_df, time_col="snapshot_time")
        t2, v2, s2 = chronological_split(mock_snapshot_df, time_col="snapshot_time")

        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(v1, v2)
        pd.testing.assert_frame_equal(s1, s2)

    def test_invalid_ratios_raise(self, mock_snapshot_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            chronological_split(
                mock_snapshot_df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2
            )

        with pytest.raises(ValueError, match="strictly positive"):
            chronological_split(
                mock_snapshot_df, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0
            )

    def test_missing_time_col_raises(self, mock_snapshot_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="non_existent_time"):
            chronological_split(mock_snapshot_df, time_col="non_existent_time")

    def test_empty_df_raises(self) -> None:
        empty = pd.DataFrame(columns=["snapshot_time"])
        with pytest.raises(ValueError, match="Cannot split an empty DataFrame"):
            chronological_split(empty, time_col="snapshot_time")

    def test_too_small_df_raises(self) -> None:
        tiny = pd.DataFrame({"snapshot_time": [pd.Timestamp("2024-01-01")]})
        with pytest.raises(ValueError, match="too small"):
            chronological_split(tiny, time_col="snapshot_time")


# ---------------------------------------------------------------------------
# 2. Feature Contract & Leakage Safeguards
# ---------------------------------------------------------------------------


class TestFeatureContractAndLeakage:
    def test_get_model_feature_names_filters_prohibited(
        self, mock_snapshot_df: pd.DataFrame
    ) -> None:
        num_cols, cat_cols = get_model_feature_names(
            mock_snapshot_df, target_col="remaining_time_minutes"
        )

        all_selected = set(num_cols + cat_cols)
        for prohibited in PROHIBITED_FEATURE_COLUMNS:
            assert prohibited not in all_selected

        assert "remaining_time_minutes" not in all_selected
        assert "age" in num_cols
        assert "acuity" in num_cols
        assert "heartrate" in num_cols
        assert "gender" in cat_cols
        assert "arrival_transport" in cat_cols

        # Test with custom prohibited set
        df_extra = mock_snapshot_df.copy()
        df_extra["extra_leak"] = 123
        num_c2, _ = get_model_feature_names(df_extra, prohibited_cols={"extra_leak"})
        assert "extra_leak" not in num_c2

    def test_xgboost_rejects_leaked_target_columns(
        self, mock_snapshot_df: pd.DataFrame
    ) -> None:
        model = XGBoostCandidate(n_estimators=10)
        y = mock_snapshot_df["remaining_time_minutes"]

        # Attempt to fit with prohibited column in X
        with pytest.raises(ValueError, match="Data leakage detected"):
            model.fit(mock_snapshot_df[["age", "outtime"]], y)

        with pytest.raises(ValueError, match="Data leakage detected"):
            model.fit(mock_snapshot_df[["age", "remaining_time_minutes"]], y)

    def test_preprocessor_training_only_fit_unseen_category(self) -> None:
        # Train on categories A, B
        train_df = pd.DataFrame(
            {
                "numeric_feat": [1.0, 2.0, 3.0],
                "cat_feat": ["A", "B", "A"],
            }
        )
        preprocessor = build_preprocessor(
            categorical_cols=["cat_feat"], numeric_cols=["numeric_feat"]
        )
        train_trans = preprocessor.fit_transform(train_df)
        assert train_trans.shape[0] == 3

        # Test set contains unseen category 'C'
        test_df = pd.DataFrame(
            {
                "numeric_feat": [4.0],
                "cat_feat": ["C"],
            }
        )
        # Should transform smoothly without error; unseen OHE columns encoded as 0
        test_trans = preprocessor.transform(test_df)
        assert test_trans.shape[0] == 1


# ---------------------------------------------------------------------------
# 3. Baseline Models Tests
# ---------------------------------------------------------------------------


class TestBaselines:
    def test_global_median_baseline(self) -> None:
        baseline = GlobalMedianBaseline()

        with pytest.raises(NotFittedError):
            baseline.predict(pd.DataFrame({"x": [1, 2]}))

        X_train = pd.DataFrame({"dummy": [1, 2, 3, 4, 5]})
        y_train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        baseline.fit(X_train, y_train)

        assert baseline.median_remaining_time_ == 30.0

        X_test = pd.DataFrame({"dummy": [99, 100, 101]})
        preds = baseline.predict(X_test)
        assert len(preds) == 3
        np.testing.assert_allclose(preds, [30.0, 30.0, 30.0])

    def test_global_median_empty_raises(self) -> None:
        baseline = GlobalMedianBaseline()
        with pytest.raises(ValueError, match="Cannot fit"):
            baseline.fit(pd.DataFrame(), np.array([]))

    def test_acuity_stratified_baseline(self) -> None:
        baseline = AcuityStratifiedMedianBaseline(acuity_col="acuity")

        with pytest.raises(NotFittedError):
            baseline.predict(pd.DataFrame({"acuity": [1]}))

        X_train = pd.DataFrame({"acuity": [1, 1, 2, 2, 3]})
        y_train = np.array(
            [60.0, 80.0, 120.0, 140.0, 300.0]
        )  # medians: 1->70, 2->130, 3->300
        baseline.fit(X_train, y_train)

        assert baseline.global_median_ == 120.0
        assert baseline.acuity_medians_[1] == 70.0
        assert baseline.acuity_medians_[2] == 130.0
        assert baseline.acuity_medians_[3] == 300.0

        # Predict with known, unseen, and NaN acuity
        X_test = pd.DataFrame({"acuity": [1, 2, 3, 5, np.nan]})
        preds = baseline.predict(X_test)

        assert preds[0] == 70.0
        assert preds[1] == 130.0
        assert preds[2] == 300.0
        # Acuity 5 and NaN should fall back safely to global median (120.0)
        assert preds[3] == 120.0
        assert preds[4] == 120.0

    def test_ridge_regression_baseline(self, mock_snapshot_df: pd.DataFrame) -> None:
        X = mock_snapshot_df[
            ["age", "acuity", "heartrate", "gender", "arrival_transport"]
        ]
        y = mock_snapshot_df["remaining_time_minutes"]

        ridge = RidgeRegressionBaseline(
            numeric_cols=["age", "acuity", "heartrate"],
            categorical_cols=["gender", "arrival_transport"],
        )
        ridge.fit(X, y)

        preds = ridge.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()

    def test_ridge_requires_dataframe(self) -> None:
        ridge = RidgeRegressionBaseline()
        with pytest.raises(TypeError, match="DataFrame"):
            ridge.fit(np.array([[1, 2]]), np.array([10.0]))


# ---------------------------------------------------------------------------
# 4. XGBoost Candidate Tests
# ---------------------------------------------------------------------------


class TestXGBoostCandidate:
    def test_fit_and_predict(self, mock_snapshot_df: pd.DataFrame) -> None:
        train_df, _, test_df = chronological_split(
            mock_snapshot_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        features = ["age", "acuity", "heartrate", "gender", "arrival_transport"]
        X_train = train_df[features]
        y_train = train_df["remaining_time_minutes"]
        X_test = test_df[features]

        model = XGBoostCandidate(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            clip_predictions_at_zero=True,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert np.all(preds >= 0.0)  # Clipped at zero
        assert model.feature_importances_ is not None

    def test_handles_missing_numerics(self, mock_snapshot_df: pd.DataFrame) -> None:
        features = ["age", "heartrate"]  # heartrate contains NaNs
        X = mock_snapshot_df[features]
        y = mock_snapshot_df["remaining_time_minutes"]

        model = XGBoostCandidate(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(X)
        assert not np.isnan(preds).any()

    def test_deterministic_output(self, mock_snapshot_df: pd.DataFrame) -> None:
        features = ["age", "acuity", "gender"]
        X = mock_snapshot_df[features]
        y = mock_snapshot_df["remaining_time_minutes"]

        m1 = XGBoostCandidate(n_estimators=15, random_state=42).fit(X, y)
        m2 = XGBoostCandidate(n_estimators=15, random_state=42).fit(X, y)

        p1 = m1.predict(X)
        p2 = m2.predict(X)

        np.testing.assert_allclose(p1, p2)


# ---------------------------------------------------------------------------
# 5. Predictor Container & Serialization Tests
# ---------------------------------------------------------------------------


class TestPredictorAndSerialization:
    def test_predict_single_and_serialization(
        self, mock_snapshot_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        features = ["age", "acuity", "heartrate", "gender"]
        X = mock_snapshot_df[features]
        y = mock_snapshot_df["remaining_time_minutes"]

        base_model = XGBoostCandidate(n_estimators=10, max_depth=2, random_state=42)
        base_model.fit(X, y)

        predictor = PatientFlowPredictor(
            model=base_model,
            model_name="xgboost_regressor",
            model_version="0.1.0",
            feature_names=features,
            numeric_cols=["age", "acuity", "heartrate"],
            categorical_cols=["gender"],
        )

        single_patient = {
            "age": 45,
            "acuity": 3,
            "heartrate": 88.0,
            "gender": "F",
        }
        res = predictor.predict_single(single_patient)

        assert "predicted_remaining_time_minutes" in res
        assert isinstance(res["predicted_remaining_time_minutes"], float)
        assert res["prediction_interval"] is None
        assert res["model_name"] == "xgboost_regressor"
        assert res["model_version"] == "0.1.0"
        assert res["features_used"] == features

        # Test Serialization
        save_file = tmp_path / "models" / "patient_time" / "model.joblib"
        saved_path = save_predictor(predictor, save_file)
        assert saved_path.is_file()

        loaded_predictor = load_predictor(saved_path)
        assert isinstance(loaded_predictor, PatientFlowPredictor)
        assert loaded_predictor.model_name == predictor.model_name

        res_loaded = loaded_predictor.predict_single(single_patient)
        assert (
            res_loaded["predicted_remaining_time_minutes"]
            == res["predicted_remaining_time_minutes"]
        )

    def test_load_non_existent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_predictor("non_existent_model_path.joblib")


# ---------------------------------------------------------------------------
# 6. Evaluation Metrics Tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_evaluate_regression_known_values(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 19.0, 31.0])
        # Errors: |+2|, |-1|, |+1|
        # MAE: (2 + 1 + 1) / 3 = 1.333333...
        # MSE: (4 + 1 + 1) / 3 = 2.0 -> RMSE = sqrt(2) ~ 1.41421...
        # MedAE: median([2, 1, 1]) = 1.0

        metrics = evaluate_regression(y_true, y_pred)
        assert pytest.approx(metrics["mae"], rel=1e-4) == 4.0 / 3.0
        assert pytest.approx(metrics["rmse"], rel=1e-4) == np.sqrt(2.0)
        assert metrics["medae"] == 1.0
        assert "r2" in metrics

    def test_evaluate_regression_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="Input length mismatch"):
            evaluate_regression([1.0, 2.0], [1.0])

    def test_evaluate_regression_empty_raise(self) -> None:
        with pytest.raises(ValueError, match="empty arrays"):
            evaluate_regression([], [])

    def test_evaluate_subgroups(self) -> None:
        df = pd.DataFrame(
            {
                "actual": [10.0, 20.0, 30.0, 40.0],
                "pred": [12.0, 18.0, 31.0, 39.0],
                "acuity": [1, 1, 2, 2],
            }
        )
        res = evaluate_subgroups(
            df, y_true_col="actual", y_pred_col="pred", subgroup_col="acuity"
        )

        assert len(res) == 2
        assert "acuity" in res.columns
        assert "sample_count" in res.columns
        assert "mae" in res.columns
        assert "rmse" in res.columns
        assert "medae" in res.columns
        assert "r2" in res.columns
        assert list(res["sample_count"]) == [2, 2]

    def test_format_metrics_summary(self) -> None:
        metrics = {"mae": 15.2, "rmse": 20.4, "medae": 12.0, "r2": 0.4567}
        formatted = format_metrics_summary(metrics, prefix="[Test]")
        assert "[Test] MAE:   15.20 min" in formatted
        assert "[Test] RMSE:  20.40 min" in formatted
        assert "[Test] MedAE: 12.00 min" in formatted
        assert "[Test] R²:    0.4567" in formatted

    def test_evaluate_subgroups_missing_column_raises(self) -> None:
        df = pd.DataFrame({"actual": [1.0], "pred": [1.0]})
        with pytest.raises(KeyError, match="missing_col"):
            evaluate_subgroups(df, "actual", "pred", "missing_col")

    def test_evaluate_subgroups_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty DataFrame"):
            evaluate_subgroups(pd.DataFrame(), "a", "b", "c")

    def test_evaluate_regression_all_nans_raises(self) -> None:
        with pytest.raises(ValueError, match="All pairs contain NaN"):
            evaluate_regression([np.nan], [np.nan])

    def test_evaluate_regression_single_sample_r2_zero(self) -> None:
        res = evaluate_regression([10.0], [12.0])
        assert res["mae"] == 2.0
        assert res["r2"] == 0.0


# ---------------------------------------------------------------------------
# 7. Edge Cases & Boundary Handling Tests
# ---------------------------------------------------------------------------


class TestModelEdgeCases:
    def test_baseline_acuity_missing_col_raises(self) -> None:
        baseline = AcuityStratifiedMedianBaseline(acuity_col="acuity")
        with pytest.raises(KeyError, match="Acuity column 'acuity' not found"):
            baseline.fit(pd.DataFrame({"wrong_col": [1, 2]}), [10.0, 20.0])

    def test_baseline_acuity_all_nans_raises(self) -> None:
        baseline = AcuityStratifiedMedianBaseline()
        with pytest.raises(ValueError, match="contains only NaN"):
            baseline.fit(pd.DataFrame({"acuity": [1, 2]}), [np.nan, np.nan])

    def test_baseline_acuity_ndarray_input(self) -> None:
        baseline = AcuityStratifiedMedianBaseline()
        X_arr = np.array([[1], [2], [1]])
        baseline.fit(X_arr, [50.0, 100.0, 60.0])
        preds = baseline.predict(np.array([[1], [2], [3]]))
        assert preds[0] == 55.0
        assert preds[1] == 100.0
        assert preds[2] == 60.0  # fallback to global median

    def test_baseline_acuity_predict_missing_col_raises(self) -> None:
        baseline = AcuityStratifiedMedianBaseline()
        baseline.fit(pd.DataFrame({"acuity": [1, 2]}), [50.0, 100.0])
        with pytest.raises(KeyError, match="Acuity column 'acuity' not found"):
            baseline.predict(pd.DataFrame({"wrong": [1]}))

    def test_ridge_unfitted_raises(self) -> None:
        ridge = RidgeRegressionBaseline()
        with pytest.raises(NotFittedError):
            ridge.predict(pd.DataFrame({"a": [1]}))

    def test_ridge_predict_requires_dataframe(self) -> None:
        ridge = RidgeRegressionBaseline()
        ridge.fit(pd.DataFrame({"a": [1, 2]}), [1.0, 2.0])
        with pytest.raises(TypeError, match="DataFrame"):
            ridge.predict(np.array([[1]]))

    def test_xgboost_unfitted_raises(self) -> None:
        xgb_m = XGBoostCandidate()
        with pytest.raises(NotFittedError):
            xgb_m.predict(pd.DataFrame({"a": [1]}))
        with pytest.raises(NotFittedError):
            _ = xgb_m.feature_importances_

    def test_xgboost_fit_requires_dataframe(self) -> None:
        xgb_m = XGBoostCandidate()
        with pytest.raises(TypeError, match="DataFrame"):
            xgb_m.fit(np.array([[1, 2]]), [1.0, 2.0])

    def test_xgboost_fit_empty_y_raises(self) -> None:
        xgb_m = XGBoostCandidate()
        with pytest.raises(ValueError, match="empty target array"):
            xgb_m.fit(pd.DataFrame({"a": []}), [])

    def test_xgboost_predict_requires_dataframe(
        self, mock_snapshot_df: pd.DataFrame
    ) -> None:
        xgb_m = XGBoostCandidate(n_estimators=5)
        xgb_m.fit(mock_snapshot_df[["age"]], mock_snapshot_df["remaining_time_minutes"])
        with pytest.raises(TypeError, match="DataFrame"):
            xgb_m.predict(np.array([[50]]))

    def test_xgboost_predict_rejects_leaked_columns(
        self, mock_snapshot_df: pd.DataFrame
    ) -> None:
        xgb_m = XGBoostCandidate(n_estimators=5)
        xgb_m.fit(mock_snapshot_df[["age"]], mock_snapshot_df["remaining_time_minutes"])
        with pytest.raises(ValueError, match="Data leakage detected"):
            xgb_m.predict(mock_snapshot_df[["age", "outtime"]])

    def test_xgboost_with_eval_set(self, mock_snapshot_df: pd.DataFrame) -> None:
        train, val, _ = chronological_split(
            mock_snapshot_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        xgb_m = XGBoostCandidate(n_estimators=10)
        xgb_m.fit(
            train[["age", "acuity"]],
            train["remaining_time_minutes"],
            eval_set=[(val[["age", "acuity"]], val["remaining_time_minutes"])],
        )
        preds = xgb_m.predict(val[["age", "acuity"]])
        assert len(preds) == len(val)

    def test_predictor_missing_required_features_raises(
        self, mock_snapshot_df: pd.DataFrame
    ) -> None:
        base_model = XGBoostCandidate(n_estimators=5)
        base_model.fit(
            mock_snapshot_df[["age", "acuity"]],
            mock_snapshot_df["remaining_time_minutes"],
        )
        predictor = PatientFlowPredictor(
            model=base_model,
            model_name="test_model",
            feature_names=["age", "acuity"],
        )
        with pytest.raises(ValueError, match="missing required feature columns"):
            predictor.predict(pd.DataFrame({"age": [50]}))

    def test_predictor_rejects_non_dataframe(self) -> None:
        predictor = PatientFlowPredictor(model=None, model_name="test")
        with pytest.raises(TypeError, match="pandas DataFrame"):
            predictor.predict([[1, 2]])

    def test_predictor_rejects_leaked_target_columns(
        self, mock_snapshot_df: pd.DataFrame
    ) -> None:
        predictor = PatientFlowPredictor(model=None, model_name="test")
        with pytest.raises(ValueError, match="Data leakage detected"):
            predictor.predict(mock_snapshot_df[["age", "remaining_time_minutes"]])

    def test_predictor_model_without_predict_raises(self) -> None:
        predictor = PatientFlowPredictor(model=object(), model_name="test")
        with pytest.raises(NotFittedError, match="does not implement a predict method"):
            predictor.predict(pd.DataFrame({"a": [1]}))

    def test_load_predictor_invalid_type_raises(self, tmp_path: Path) -> None:
        dummy_file = tmp_path / "not_a_predictor.joblib"
        import joblib

        joblib.dump({"not": "a predictor"}, dummy_file)
        with pytest.raises(
            TypeError, match="Expected loaded object to be PatientFlowPredictor"
        ):
            load_predictor(dummy_file)

    def test_build_preprocessor_with_scaling(self) -> None:
        prep = build_preprocessor(
            categorical_cols=["gender"],
            numeric_cols=["age", "heartrate"],
            scale_numeric=True,
        )
        df = pd.DataFrame(
            {"gender": ["M", "F"], "age": [25.0, 50.0], "heartrate": [70.0, np.nan]}
        )
        res = prep.fit_transform(df)
        assert res.shape == (2, 4)  # 2 numeric + 2 OHE categories
