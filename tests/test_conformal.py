"""Unit tests for QueueMind split-conformal prediction uncertainty module."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from queuemind.models.conformal import ConformalIntervalCalibrator
from queuemind.models.evaluate import (
    evaluate_interval_subgroups,
    evaluate_prediction_intervals,
)
from queuemind.models.predict import (
    PatientFlowPredictor,
    load_predictor,
    save_predictor,
)
from queuemind.models.train import XGBoostCandidate, chronological_split


@pytest.fixture
def mock_cal_test_data() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, XGBoostCandidate]
):
    """Generate chronological train, val, and test data with a fitted model."""
    n = 100
    base_time = pd.Timestamp("2024-01-01 08:00:00")
    times = [base_time + pd.Timedelta(minutes=15 * i) for i in range(n)]

    rng = np.random.RandomState(42)
    acuities = rng.choice([1, 2, 3, 4, 5], size=n)
    ages = rng.randint(18, 90, size=n)
    heartrates = rng.uniform(60, 130, size=n)
    genders = rng.choice(["M", "F"], size=n)
    remaining = (acuities * 30.0) + (ages * 0.5) + rng.uniform(-10, 10, size=n)
    remaining = np.maximum(5.0, remaining)

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
            "disposition": rng.choice(["HOME", "ADMITTED"], size=n),
            "remaining_time_minutes": remaining,
        }
    )

    train_df, val_df, test_df = chronological_split(
        df, time_col="snapshot_time", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    features = ["acuity", "age", "heartrate", "gender"]
    model = XGBoostCandidate(
        n_estimators=20,
        max_depth=3,
        random_state=42,
        numeric_cols=["acuity", "age", "heartrate"],
        categorical_cols=["gender"],
    )
    model.fit(train_df[features], train_df["remaining_time_minutes"])

    return train_df, val_df, test_df, model


class TestConformalIntervalCalibrator:
    def test_calibration_and_cutoffs(self) -> None:
        calibrator = ConformalIntervalCalibrator(default_coverage_level=0.90)
        assert not calibrator.is_calibrated

        # Synthetic residuals: [10, 20, 30, ..., 100]
        y_true = np.arange(10, 110, 10, dtype=float)
        y_pred = np.zeros(10, dtype=float)
        calibrator.calibrate(y_true, y_pred)

        assert calibrator.is_calibrated
        assert calibrator.n_calibration_samples_ == 10

        q80 = calibrator.get_cutoff(coverage_level=0.80)
        q90 = calibrator.get_cutoff(coverage_level=0.90)
        q95 = calibrator.get_cutoff(coverage_level=0.95)

        assert q80 <= q90 <= q95
        assert q90 > 0.0

    def test_predict_interval_bounds_and_non_negativity(self) -> None:
        calibrator = ConformalIntervalCalibrator(default_coverage_level=0.90)
        calibrator.calibrate(np.array([10.0, 20.0, 30.0]), np.array([8.0, 22.0, 33.0]))

        # Large prediction: lower and upper bounds around prediction
        lower, upper = calibrator.predict_interval(np.array([50.0]))
        assert lower[0] <= 50.0 <= upper[0]
        assert lower[0] >= 0.0

        # Small prediction: lower bound should be clipped cleanly at zero
        lower_small, _ = calibrator.predict_interval(np.array([1.0]))
        assert lower_small[0] == 0.0

    def test_get_interval_for_prediction_schema(self) -> None:
        calibrator = ConformalIntervalCalibrator(default_coverage_level=0.90)
        calibrator.calibrate(np.array([10.0, 20.0]), np.array([12.0, 18.0]))

        res = calibrator.get_interval_for_prediction(45.0, coverage_level=0.90)
        assert "lower_minutes" in res
        assert "upper_minutes" in res
        assert res["coverage_level"] == 0.90
        assert res["method"] == "split_conformal"
        assert res["lower_minutes"] <= 45.0 <= res["upper_minutes"]

    def test_invalid_coverage_level_raises(self) -> None:
        calibrator = ConformalIntervalCalibrator()
        calibrator.calibrate([1.0], [1.0])

        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            calibrator.get_cutoff(0.0)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            calibrator.get_cutoff(1.0)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            calibrator.get_cutoff(1.2)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            calibrator.get_cutoff(-0.1)

    def test_uncalibrated_raises(self) -> None:
        calibrator = ConformalIntervalCalibrator()
        with pytest.raises(NotFittedError, match="not calibrated"):
            calibrator.get_cutoff(0.90)
        with pytest.raises(NotFittedError, match="not calibrated"):
            calibrator.predict_interval(50.0)

    def test_calibration_input_validation(self) -> None:
        calibrator = ConformalIntervalCalibrator()

        with pytest.raises(ValueError, match="empty"):
            calibrator.calibrate([], [])

        with pytest.raises(ValueError, match="Length mismatch"):
            calibrator.calibrate([1.0, 2.0], [1.0])

        with pytest.raises(ValueError, match="NaN"):
            calibrator.calibrate([np.nan, 2.0], [1.0, 2.0])


class TestIntervalEvaluation:
    def test_evaluate_prediction_intervals_known_values(self) -> None:
        # 10 samples: 8 inside interval, 2 outside
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        # Width is constant: 20 (from pred-10 to pred+10)
        low = y_true - 10.0
        high = y_true + 10.0
        # Make samples 0 and 9 fall outside
        low[0] = 15.0  # true 10 is below lower 15
        high[9] = 95.0  # true 100 is above upper 95

        res = evaluate_prediction_intervals(y_true, low, high)
        assert pytest.approx(res["empirical_coverage"], rel=1e-4) == 0.80
        assert res["mean_width"] > 0.0
        assert res["median_width"] > 0.0

    def test_evaluate_prediction_intervals_errors(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            evaluate_prediction_intervals([], [], [])

        with pytest.raises(ValueError, match="mismatch"):
            evaluate_prediction_intervals([1.0], [1.0], [1.0, 2.0])

        with pytest.raises(
            ValueError, match="All interval evaluation pairs contain NaN"
        ):
            evaluate_prediction_intervals([np.nan], [np.nan], [np.nan])

    def test_evaluate_interval_subgroups(self) -> None:
        df = pd.DataFrame(
            {
                "y_true": [10.0, 20.0, 30.0, 40.0],
                "low": [5.0, 15.0, 25.0, 35.0],
                "high": [15.0, 25.0, 35.0, 45.0],
                "acuity": [1, 1, 2, 2],
            }
        )
        res = evaluate_interval_subgroups(
            df,
            y_true_col="y_true",
            lower_col="low",
            upper_col="high",
            subgroup_col="acuity",
        )
        assert len(res) == 2
        assert "empirical_coverage" in res.columns
        assert "mean_width" in res.columns
        assert list(res["empirical_coverage"]) == [1.0, 1.0]

    def test_evaluate_interval_subgroups_missing_col_raises(self) -> None:
        df = pd.DataFrame({"a": [1.0], "b": [1.0]})
        with pytest.raises(KeyError, match="Required column"):
            evaluate_interval_subgroups(df, "a", "b", "c", "missing")

    def test_evaluate_interval_subgroups_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty DataFrame"):
            evaluate_interval_subgroups(pd.DataFrame(), "a", "b", "c", "d")


class TestPredictorConformalIntegration:
    def test_predictor_with_calibrator(
        self,
        mock_cal_test_data: tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame, XGBoostCandidate
        ],
        tmp_path: Path,
    ) -> None:
        train_df, val_df, test_df, model = mock_cal_test_data
        features = ["acuity", "age", "heartrate", "gender"]

        # 1. Calibrate on validation set (strictly separate from test set)
        val_preds = model.predict(val_df[features])
        calibrator = ConformalIntervalCalibrator(default_coverage_level=0.90)
        calibrator.calibrate(val_df["remaining_time_minutes"], val_preds)

        # 2. Attach calibrator to PatientFlowPredictor
        predictor = PatientFlowPredictor(
            model=model,
            model_name="xgb_conformal",
            model_version="0.1.0",
            feature_names=features,
            calibrator=calibrator,
        )

        single_patient = test_df.iloc[0][features].to_dict()
        res = predictor.predict_single(single_patient, coverage_level=0.90)

        assert res["predicted_remaining_time_minutes"] > 0.0
        assert res["prediction_interval"] is not None

        interval = res["prediction_interval"]
        assert interval["coverage_level"] == 0.90
        assert interval["method"] == "split_conformal"
        assert (
            interval["lower_minutes"]
            <= res["predicted_remaining_time_minutes"]
            <= interval["upper_minutes"]
        )

        # 3. Test Serialization & Deserialization
        save_file = tmp_path / "models" / "patient_time" / "calibrated_model.joblib"
        save_predictor(predictor, save_file)

        loaded = load_predictor(save_file)
        loaded_res = loaded.predict_single(single_patient, coverage_level=0.90)

        assert (
            loaded_res["predicted_remaining_time_minutes"]
            == res["predicted_remaining_time_minutes"]
        )
        assert (
            loaded_res["prediction_interval"]["lower_minutes"]
            == interval["lower_minutes"]
        )
        assert (
            loaded_res["prediction_interval"]["upper_minutes"]
            == interval["upper_minutes"]
        )

    def test_predictor_without_calibrator_returns_none_interval(
        self,
        mock_cal_test_data: tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame, XGBoostCandidate
        ],
    ) -> None:
        _, _, test_df, model = mock_cal_test_data
        features = ["acuity", "age", "heartrate", "gender"]

        # Predictor initialized WITHOUT calibrator (backward compatibility)
        predictor = PatientFlowPredictor(
            model=model,
            model_name="xgb_uncalibrated",
            feature_names=features,
        )

        single_patient = test_df.iloc[0][features].to_dict()
        res = predictor.predict_single(single_patient)

        assert res["predicted_remaining_time_minutes"] > 0.0
        # Backward compatibility: prediction_interval is None when uncalibrated
        assert res["prediction_interval"] is None
