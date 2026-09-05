"""Tests for Emergency Department congestion forecasting and bottleneck intelligence."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from queuemind.features.bottleneck_features import (
    CongestionLevel,
    classify_congestion_state,
    detect_bottleneck_indicators,
)
from queuemind.features.time_grid import (
    create_congestion_targets,
    extract_department_snapshot_features,
    generate_congestion_dataset,
    generate_time_grid,
)
from queuemind.models.conformal import ConformalIntervalCalibrator
from queuemind.models.congestion import (
    CongestionPredictor,
    LastValueCongestionBaseline,
    RidgeCongestionModel,
    TimeOfDayMedianCongestionBaseline,
    XGBoostCongestionModel,
    filter_congestion_features,
    temporal_congestion_split,
)
from queuemind.models.evaluate import (
    evaluate_congestion_forecasts,
    format_congestion_metrics_summary,
)


@pytest.fixture
def mock_ed_encounters() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a synthetic cohort of ED encounters spanning 48 hours."""
    n = 120
    base_time = pd.Timestamp("2024-01-01 00:00:00")
    rng = np.random.RandomState(42)

    intimes = [
        base_time + timedelta(minutes=int(rng.randint(0, 2400))) for _ in range(n)
    ]
    # Durations between 60 and 480 minutes
    durations = [timedelta(minutes=int(rng.randint(60, 480))) for _ in range(n)]
    outtimes = [in_t + dur for in_t, dur in zip(intimes, durations)]

    edstays = pd.DataFrame(
        {
            "stay_id": [1000 + i for i in range(n)],
            "subject_id": [2000 + i for i in range(n)],
            "intime": intimes,
            "outtime": outtimes,
            "gender": rng.choice(["M", "F"], size=n),
            "arrival_transport": rng.choice(["WALK IN", "AMBULANCE"], size=n),
            "disposition": rng.choice(["HOME", "ADMITTED"], size=n),
        }
    )

    triage = pd.DataFrame(
        {
            "stay_id": [1000 + i for i in range(n)],
            "acuity": rng.choice([1, 2, 3, 4, 5], size=n),
        }
    )

    return edstays, triage


class TestTimeGrid:
    """Test suite for time-grid generation and feature extraction."""

    def test_generate_time_grid_spacing_and_bounds(self) -> None:
        start = pd.Timestamp("2024-01-01 08:00:00")
        end = pd.Timestamp("2024-01-01 10:00:00")
        grid = generate_time_grid(start, end, step_minutes=30)

        assert len(grid) == 5
        assert grid[0] == start
        assert grid[-1] == end
        assert (grid[1] - grid[0]).total_seconds() == 1800

    def test_generate_time_grid_invalid_step_or_range_raises(self) -> None:
        start = pd.Timestamp("2024-01-01 10:00:00")
        end = pd.Timestamp("2024-01-01 08:00:00")

        with pytest.raises(ValueError, match="step_minutes must be positive"):
            generate_time_grid(start, end, step_minutes=0)

        with pytest.raises(ValueError, match="cannot be strictly after"):
            generate_time_grid(start, end, step_minutes=15)

    def test_extract_department_snapshot_features_bounds(
        self, mock_ed_encounters: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        edstays, triage = mock_ed_encounters
        snap = pd.Timestamp("2024-01-01 12:00:00")

        feats = extract_department_snapshot_features(snap, edstays, triage)

        assert "current_active_census" in feats
        assert "recent_arrivals_15m" in feats
        assert "recent_departures_60m" in feats
        assert "net_flow_60m" in feats
        assert "high_acuity_ratio" in feats
        assert feats["current_active_census"] >= 0
        assert feats["snapshot_hour"] == 12
        assert 0.0 <= feats["high_acuity_ratio"] <= 1.0

    def test_extract_department_snapshot_features_missing_cols_fallback(self) -> None:
        empty_df = pd.DataFrame()
        feats = extract_department_snapshot_features("2024-01-01 12:00:00", empty_df)
        assert feats["current_active_census"] == 0
        assert feats["net_flow_60m"] == 0

    def test_create_congestion_targets(
        self, mock_ed_encounters: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        edstays, _ = mock_ed_encounters
        snap = pd.Timestamp("2024-01-01 12:00:00")
        targets = create_congestion_targets(snap, edstays, horizons=(30, 60, 120))

        assert "target_census_30m" in targets
        assert "target_census_60m" in targets
        assert "target_census_120m" in targets
        for val in targets.values():
            assert val >= 0

    def test_generate_congestion_dataset_alignment(
        self, mock_ed_encounters: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        edstays, triage = mock_ed_encounters
        X, Y = generate_congestion_dataset(
            edstays,
            triage,
            step_minutes=60,
            horizons=(30, 60, 120),
        )

        assert len(X) == len(Y)
        assert len(X) > 0
        assert "current_active_census" in X.columns
        assert "target_census_30m" in Y.columns
        assert "target_census_60m" in Y.columns
        assert "target_census_120m" in Y.columns

    def test_generate_congestion_dataset_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot generate congestion dataset"):
            generate_congestion_dataset(pd.DataFrame())

    def test_future_event_leakage_invariance(
        self, mock_ed_encounters: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        edstays, triage = mock_ed_encounters
        t_snap = pd.Timestamp("2024-01-01 12:00:00")

        # Baseline snapshot features
        base_feats = extract_department_snapshot_features(t_snap, edstays, triage)

        # Inject future arrivals and departures strictly after t_snap
        future_patient = pd.DataFrame(
            [
                {
                    "stay_id": 99999,
                    "subject_id": 88888,
                    "intime": t_snap + timedelta(minutes=45),
                    "outtime": t_snap + timedelta(minutes=180),
                    "gender": "M",
                    "arrival_transport": "AMBULANCE",
                    "disposition": "ADMITTED",
                }
            ]
        )
        augmented_edstays = pd.concat([edstays, future_patient], ignore_index=True)

        new_feats = extract_department_snapshot_features(
            t_snap, augmented_edstays, triage
        )

        # Assert 100% invariance: future patient cannot alter present features
        assert base_feats == new_feats


class TestTemporalCongestionSplit:
    """Test suite for chronological splitting with temporal embargo."""

    def test_temporal_congestion_split_embargo_gap(
        self, mock_ed_encounters: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        edstays, triage = mock_ed_encounters
        X, _ = generate_congestion_dataset(
            edstays, triage, step_minutes=15, horizons=(30, 60, 120)
        )

        train_df, val_df, test_df = temporal_congestion_split(
            X,
            time_col="snapshot_time",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            embargo_minutes=120,
        )

        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        max_train = train_df["snapshot_time"].max()
        min_val = val_df["snapshot_time"].min()
        max_val = val_df["snapshot_time"].max()
        min_test = test_df["snapshot_time"].min()

        embargo_delta = timedelta(minutes=120)

        # Strict embargo guarantees:
        assert max_train + embargo_delta <= min_val
        assert max_val + embargo_delta <= min_test

    def test_temporal_congestion_split_invalid_ratios_raises(self) -> None:
        df = pd.DataFrame({"snapshot_time": [pd.Timestamp("2024-01-01")] * 10})
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            temporal_congestion_split(
                df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.1
            )

    def test_temporal_congestion_split_too_short_raises(self) -> None:
        times = [
            pd.Timestamp("2024-01-01 00:00:00") + timedelta(minutes=10 * i)
            for i in range(5)
        ]
        df = pd.DataFrame({"snapshot_time": times})
        with pytest.raises(ValueError, match="too short for split with"):
            temporal_congestion_split(df, embargo_minutes=120)


class TestCongestionBaselines:
    """Test suite for operational baselines."""

    def test_last_value_baseline(self) -> None:
        X = pd.DataFrame(
            {
                "current_active_census": [20, 35, 42],
                "snapshot_hour": [8, 12, 16],
            }
        )
        baseline = LastValueCongestionBaseline(horizons=(30, 60, 120))
        baseline.fit(X)

        preds = baseline.predict(X)

        assert "pred_census_30m" in preds.columns
        assert "pred_census_60m" in preds.columns
        assert "pred_census_120m" in preds.columns
        assert np.array_equal(preds["pred_census_30m"].values, [20.0, 35.0, 42.0])
        assert np.array_equal(preds["pred_census_120m"].values, [20.0, 35.0, 42.0])

    def test_last_value_baseline_unfitted_raises(self) -> None:
        baseline = LastValueCongestionBaseline()
        with pytest.raises(NotFittedError):
            baseline.predict(pd.DataFrame({"current_active_census": [10]}))

    def test_time_of_day_baseline(self) -> None:
        times = [
            pd.Timestamp("2024-01-01 08:00:00"),  # Monday 8am
            pd.Timestamp("2024-01-01 08:30:00"),  # Monday 8am
            pd.Timestamp("2024-01-08 08:00:00"),  # Monday 8am
        ]
        X = pd.DataFrame(
            {
                "snapshot_time": times,
                "current_active_census": [20, 24, 22],
                "snapshot_hour": [8, 8, 8],
                "snapshot_day_of_week": [0, 0, 0],
            }
        )
        model = TimeOfDayMedianCongestionBaseline(horizons=(30, 60))
        model.fit(X)

        preds = model.predict(X)
        assert len(preds) == 3
        # Forecasts predict the median
        assert preds["pred_census_30m"].iloc[0] > 0


class TestCongestionModels:
    """Test suite for Ridge and XGBoost congestion forecasters."""

    @pytest.fixture
    def mock_train_test(
        self, mock_ed_encounters: tuple[pd.DataFrame, pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        edstays, triage = mock_ed_encounters
        X, Y = generate_congestion_dataset(
            edstays, triage, step_minutes=30, horizons=(30, 60, 120)
        )
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]
        return X_train, Y_train, X_test, Y_test

    def test_ridge_congestion_model(
        self,
        mock_train_test: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        X_train, Y_train, X_test, Y_test = mock_train_test
        model = RidgeCongestionModel(horizons=(30, 60, 120))
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        assert len(preds) == len(X_test)
        for h in (30, 60, 120):
            col = f"pred_census_{h}m"
            assert col in preds.columns
            # Non-negative projection
            assert (preds[col] >= 0.0).all()

    def test_xgboost_congestion_model(
        self,
        mock_train_test: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        X_train, Y_train, X_test, Y_test = mock_train_test
        model = XGBoostCongestionModel(
            horizons=(30, 60, 120), n_estimators=20, max_depth=3
        )
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        assert len(preds) == len(X_test)
        for h in (30, 60, 120):
            col = f"pred_census_{h}m"
            assert col in preds.columns
            assert (preds[col] >= 0.0).all()

        importances = model.get_feature_importances(horizon=60)
        assert len(importances) > 0
        assert sum(importances.values()) > 0.0

    def test_filter_congestion_features(self) -> None:
        df = pd.DataFrame(
            {
                "snapshot_time": ["2024-01-01"],
                "target_census_30m": [25],
                "target_census_60m": [30],
                "current_active_census": [20],
                "net_flow_60m": [5],
            }
        )
        clean = filter_congestion_features(df)
        assert "snapshot_time" not in clean.columns
        assert "target_census_30m" not in clean.columns
        assert "current_active_census" in clean.columns
        assert "net_flow_60m" in clean.columns


class TestCongestionEvaluation:
    """Test suite for multi-horizon forecast evaluation."""

    def test_evaluate_congestion_forecasts(self) -> None:
        y_true = pd.DataFrame(
            {
                "target_census_30m": [20.0, 30.0, 40.0],
                "target_census_60m": [22.0, 32.0, 42.0],
            }
        )
        y_pred = pd.DataFrame(
            {
                "pred_census_30m": [22.0, 28.0, 41.0],
                "pred_census_60m": [20.0, 35.0, 45.0],
            }
        )

        res = evaluate_congestion_forecasts(y_true, y_pred, horizons=(30, 60))

        assert "30m" in res
        assert "60m" in res
        assert res["30m"]["mae"] > 0.0
        assert res["60m"]["rmse"] > 0.0

    def test_format_congestion_metrics_summary(self) -> None:
        metrics = {
            "Persistence": {
                "30m": {"mae": 3.2, "rmse": 4.1, "medae": 2.5, "r2": 0.85},
                "60m": {"mae": 5.1, "rmse": 6.3, "medae": 4.0, "r2": 0.72},
            },
            "XGBoost": {
                "30m": {"mae": 2.1, "rmse": 2.9, "medae": 1.8, "r2": 0.92},
                "60m": {"mae": 3.4, "rmse": 4.5, "medae": 2.9, "r2": 0.83},
            },
        }

        table_str = format_congestion_metrics_summary(metrics, horizons=(30, 60))
        assert "Persistence" in table_str
        assert "XGBoost" in table_str
        assert "30m" in table_str
        assert "60m" in table_str


class TestBottleneckAndState:
    """Test suite for operational bottleneck indicators and state classification."""

    def test_classify_congestion_state(self) -> None:
        assert classify_congestion_state(15)["state"] == CongestionLevel.HEALTHY.value
        assert classify_congestion_state(35)["state"] == CongestionLevel.MODERATE.value
        assert classify_congestion_state(55)["state"] == CongestionLevel.BUSY.value
        assert classify_congestion_state(80)["state"] == CongestionLevel.CRITICAL.value

        with pytest.raises(ValueError, match="cannot be negative"):
            classify_congestion_state(-5)

    def test_detect_bottleneck_indicators_active(self) -> None:
        high_pressure_feats = {
            "current_active_census": 45,
            "net_flow_15m": 2,
            "net_flow_30m": 4,
            "net_flow_60m": 7,
            "recent_arrivals_60m": 12.0,
            "recent_departures_60m": 1.0,
            "high_acuity_ratio": 0.50,
        }

        res = detect_bottleneck_indicators(high_pressure_feats)

        assert res["indicators"]["rising_census_velocity"] is True
        assert res["indicators"]["high_arrival_pressure"] is True
        assert res["indicators"]["low_departure_throughput"] is True
        assert res["indicators"]["sustained_positive_net_flow"] is True
        assert res["indicators"]["acuity_concentration"] is True
        assert res["severity_score"] == 5
        assert len(res["active_indicators"]) == 5

    def test_detect_bottleneck_indicators_quiescent(self) -> None:
        quiet_feats = {
            "current_active_census": 12,
            "net_flow_15m": -1,
            "net_flow_30m": -2,
            "net_flow_60m": -3,
            "recent_arrivals_60m": 2.0,
            "recent_departures_60m": 5.0,
            "high_acuity_ratio": 0.15,
        }

        res = detect_bottleneck_indicators(quiet_feats)

        assert res["severity_score"] == 0
        assert len(res["active_indicators"]) == 0
        assert "within standard operational variance" in res["summary"]


class TestCongestionPredictorAndUncertainty:
    """Test suite for CongestionPredictor container and conformal intervals."""

    def test_predictor_single_and_serialization(
        self, mock_ed_encounters: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        edstays, triage = mock_ed_encounters
        X, Y = generate_congestion_dataset(
            edstays, triage, step_minutes=30, horizons=(30, 60, 120)
        )

        model = XGBoostCongestionModel(
            horizons=(30, 60, 120), n_estimators=10, max_depth=3
        )
        model.fit(X, Y)

        # Calibrate a conformal calibrator on 60m horizon
        calibrator_60m = ConformalIntervalCalibrator(default_coverage_level=0.90)
        y_true_60m = Y["target_census_60m"].values
        y_pred_60m = model.predict(X)["pred_census_60m"].values
        calibrator_60m.calibrate(y_true_60m, y_pred_60m)

        calibrators = {60: calibrator_60m}

        predictor = CongestionPredictor(
            model=model,
            horizons=(30, 60, 120),
            calibrators=calibrators,
            model_name="test_xgb_congestion",
        )

        # Predict single snapshot
        single_snapshot = X.iloc[0].to_dict()
        res = predictor.predict_single(single_snapshot)

        assert res["current_active_census"] >= 0
        assert "30m" in res["forecasts"]
        assert "60m" in res["forecasts"]
        assert "120m" in res["forecasts"]

        # 60m horizon should have calibrated prediction interval
        interval_60m = res["forecasts"]["60m"]["prediction_interval"]
        assert interval_60m is not None
        assert interval_60m["lower_minutes"] >= 0.0  # Non-negative lower bound
        assert interval_60m["upper_minutes"] >= interval_60m["lower_minutes"]

        # 30m horizon had no calibrator attached -> None
        assert res["forecasts"]["30m"]["prediction_interval"] is None

        # Serialization test
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "congestion_predictor.joblib"
            predictor.save(save_path)

            loaded = CongestionPredictor.load(save_path)
            assert loaded.model_name == "test_xgb_congestion"
            assert loaded.horizons == (30, 60, 120)
            assert 60 in loaded.calibrators
            assert loaded.calibrators[60].is_calibrated is True

            loaded_res = loaded.predict_single(single_snapshot)
            assert loaded_res["forecasts"]["60m"]["predicted_census"] == (
                res["forecasts"]["60m"]["predicted_census"]
            )

    def test_load_predictor_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            CongestionPredictor.load("non_existent_file.joblib")

    def test_load_predictor_invalid_type_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "invalid.joblib"
            import joblib

            joblib.dump({"not": "a_predictor"}, save_path)
            with pytest.raises(
                TypeError, match="Loaded object is not a CongestionPredictor"
            ):
                CongestionPredictor.load(save_path)

    def test_model_edge_cases_and_validations(self) -> None:
        # LastValueBaseline checks
        lv = LastValueCongestionBaseline()
        with pytest.raises(TypeError, match="X must be a pd.DataFrame"):
            lv.fit(np.array([1, 2, 3]))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="X must contain 'current_active_census'"):
            lv.fit(pd.DataFrame({"other": [1]}))
        lv.fit(pd.DataFrame({"current_active_census": [1]}))
        with pytest.raises(ValueError, match="X must contain 'current_active_census'"):
            lv.predict(pd.DataFrame({"other": [1]}))

        # TimeOfDayMedianBaseline checks
        tod = TimeOfDayMedianCongestionBaseline()
        with pytest.raises(TypeError, match="X must be a pd.DataFrame"):
            tod.fit("invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="X must contain 'current_active_census'"):
            tod.fit(pd.DataFrame({"snapshot_time": [pd.Timestamp("2024-01-01")]}))
        with pytest.raises(ValueError, match="X must contain 'snapshot_time'"):
            tod.fit(pd.DataFrame({"current_active_census": [10]}))
        with pytest.raises(NotFittedError):
            tod.predict(pd.DataFrame())

        # Ridge checks
        ridge = RidgeCongestionModel()
        with pytest.raises(TypeError, match="X must be a pd.DataFrame"):
            ridge.fit("not_df", pd.DataFrame())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="y must be a pd.DataFrame"):
            ridge.fit(
                pd.DataFrame({"current_active_census": [1]}),
                "not_df",  # type: ignore[arg-type]
            )
        with pytest.raises(
            KeyError, match="Target column 'target_census_30m' not found"
        ):
            ridge.fit(
                pd.DataFrame({"current_active_census": [1]}),
                pd.DataFrame({"other": [1]}),
            )
        with pytest.raises(NotFittedError):
            ridge.predict(pd.DataFrame())

        # XGBoost checks
        xgb_m = XGBoostCongestionModel()
        with pytest.raises(TypeError, match="X must be a pd.DataFrame"):
            xgb_m.fit("not_df", pd.DataFrame())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="y must be a pd.DataFrame"):
            xgb_m.fit(
                pd.DataFrame({"current_active_census": [1]}),
                "not_df",  # type: ignore[arg-type]
            )
        with pytest.raises(
            KeyError, match="Target column 'target_census_30m' not found"
        ):
            xgb_m.fit(
                pd.DataFrame({"current_active_census": [1]}),
                pd.DataFrame({"other": [1]}),
            )
        with pytest.raises(NotFittedError):
            xgb_m.predict(pd.DataFrame())
        with pytest.raises(NotFittedError):
            xgb_m.get_feature_importances(horizon=30)

        # Temporal split edge cases
        with pytest.raises(KeyError, match="Timestamp column 'missing_col' not found"):
            temporal_congestion_split(
                pd.DataFrame({"snap": [pd.Timestamp("2024-01-01")]}),
                time_col="missing_col",
            )
        with pytest.raises(ValueError, match="Split ratios must be strictly positive"):
            temporal_congestion_split(
                pd.DataFrame({"snapshot_time": [pd.Timestamp("2024-01-01")] * 10}),
                train_ratio=0.8,
                val_ratio=0.2,
                test_ratio=0.0,
            )
