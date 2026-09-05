"""Model explainability and feature attribution module for QueueMind.

This module provides interpretability utilities using TreeSHAP (SHapley Additive
exPlanations) for XGBoost patient-flow duration regression:
1. ShapExplainer: Reusable TreeSHAP explainer wrapping XGBoostCandidate and
   PatientFlowPredictor.
2. Human-readable feature mapping: Groups one-hot dummy columns back to original
   clinical and operational variables while preserving exact Shapley additivity.
3. Single-patient attribution breakdowns (top factors influencing remaining duration).
4. Department-wide global feature importance summaries.

CRITICAL SAFETY & INTERPRETABILITY DISCLAIMER:
- SHAP values explain internal model mechanics and mathematical sensitivities.
- SHAP values do NOT establish causality, biological mechanisms, or clinical etiology.
- A high positive attribution means the model associated the feature value with
  longer remaining duration; it does NOT imply that modifying this factor will
  causally alter patient stay duration.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

from queuemind.models.train import PROHIBITED_FEATURE_COLUMNS, XGBoostCandidate


class ShapExplainer:
    """TreeSHAP explainer for QueueMind XGBoost patient-flow duration models.

    Computes additive Shapley feature attributions:
        E[f(x)] + sum(phi_i) = f(x)
    and aggregates transformed one-hot columns back to original feature variables.
    """

    def __init__(self, model: Any) -> None:
        """Initialize ShapExplainer with an XGBoostCandidate or PatientFlowPredictor.

        Args:
            model: Trained XGBoostCandidate instance or PatientFlowPredictor.

        Raises:
            TypeError: If model does not wrap an XGBoost model.
            NotFittedError: If model is not yet fitted.
        """
        # Unwrap PatientFlowPredictor if passed
        candidate = getattr(model, "model", model)
        if not isinstance(candidate, XGBoostCandidate):
            cand_name = type(candidate).__name__
            raise TypeError(f"ShapExplainer requires XGBoostCandidate; got {cand_name}")

        if candidate.regressor_ is None or candidate.preprocessor_ is None:
            raise NotFittedError(
                "Cannot initialize ShapExplainer with an unfitted XGBoostCandidate."
            )

        self.regressor_ = candidate.regressor_
        self.preprocessor_: ColumnTransformer = candidate.preprocessor_
        self.feature_names_in_: list[str] = list(candidate.feature_names_in_ or [])
        self.numeric_cols_: list[str] = list(candidate.numeric_cols or [])
        self.categorical_cols_: list[str] = list(candidate.categorical_cols or [])

        # Initialize TreeExplainer
        self.explainer_ = shap.TreeExplainer(self.regressor_)
        expected_val = self.explainer_.expected_value
        self.base_value_: float = (
            float(np.ravel(expected_val)[0])
            if hasattr(expected_val, "__len__")
            else float(expected_val)
        )

        # Build reverse mapping from transformed indices to original column names
        self._build_feature_mapping()

    def _build_feature_mapping(self) -> None:
        """Map transformed preprocessor columns back to original feature names."""
        self.orig_to_transformed_indices_: dict[str, list[int]] = {
            col: [] for col in self.feature_names_in_
        }

        col_idx = 0
        for name, transformer, cols in self.preprocessor_.transformers_:
            if name == "num":
                for col in cols:
                    self.orig_to_transformed_indices_.setdefault(col, []).append(
                        col_idx
                    )
                    col_idx += 1
            elif name == "cat":
                if (
                    hasattr(transformer, "named_steps")
                    and "ohe" in transformer.named_steps
                ):
                    ohe = transformer.named_steps["ohe"]
                    for col_name, categories in zip(cols, ohe.categories_):
                        num_cats = len(categories)
                        for _ in range(num_cats):
                            self.orig_to_transformed_indices_.setdefault(
                                col_name, []
                            ).append(col_idx)
                            col_idx += 1
                else:
                    for col in cols:
                        self.orig_to_transformed_indices_.setdefault(col, []).append(
                            col_idx
                        )
                        col_idx += 1
            elif name == "remainder":
                # Handle remainder if not dropped
                pass

    def explain_single(
        self,
        snapshot: dict[str, Any] | pd.DataFrame,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        """Generate human-readable SHAP explanations for a single patient snapshot.

        Args:
            snapshot: Dictionary or single-row DataFrame of snapshot features.
            top_n: Optional limit on number of top features to return.

        Returns:
            Dictionary containing prediction, base_value, and ranked features.

        Raises:
            ValueError: If prohibited columns or multiple rows are passed.
        """
        if isinstance(snapshot, dict):
            df = pd.DataFrame([snapshot])
        elif isinstance(snapshot, pd.DataFrame):
            if len(snapshot) != 1:
                raise ValueError(
                    f"explain_single expects 1 row, got {len(snapshot)} rows."
                )
            df = snapshot.copy().reset_index(drop=True)
        else:
            raise TypeError("snapshot must be a dictionary or a single-row DataFrame.")

        # Guard against prohibited leakage
        leaked = set(df.columns).intersection(PROHIBITED_FEATURE_COLUMNS)
        if leaked:
            raise ValueError(
                "Data leakage detected! Snapshot contains prohibited columns: "
                f"{leaked}"
            )

        # Transform using training preprocessor
        X_trans = self.preprocessor_.transform(df)
        raw_shap = self.explainer_.shap_values(X_trans)

        # Handle 2D or 1D shap_values return
        if hasattr(raw_shap, "values"):
            # If Explanation object returned by newer shap
            raw_shap = raw_shap.values  # type: ignore[union-attr]

        raw_shap_1d = np.ravel(raw_shap)

        # Aggregate SHAP contributions back to original features
        feature_contributions: list[dict[str, Any]] = []
        for orig_col, indices in self.orig_to_transformed_indices_.items():
            if not indices:
                continue
            phi = float(np.sum(raw_shap_1d[indices]))
            raw_val = df[orig_col].iloc[0] if orig_col in df.columns else None

            # Convert numpy types to native Python types for clean serialization
            if hasattr(raw_val, "item"):
                raw_val = raw_val.item()  # type: ignore[union-attr]
            elif pd.isna(raw_val):
                raw_val = None

            if phi > 1e-5:
                direction = "increases_prediction"
            elif phi < -1e-5:
                direction = "decreases_prediction"
            else:
                direction = "neutral"

            feature_contributions.append(
                {
                    "name": orig_col,
                    "value": raw_val,
                    "shap_value": round(phi, 3),
                    "direction": direction,
                    "_abs_phi": abs(phi),
                }
            )

        # Rank features by absolute impact descending
        feature_contributions.sort(key=lambda x: x["_abs_phi"], reverse=True)
        for rank, item in enumerate(feature_contributions, start=1):
            item["rank"] = rank
            del item["_abs_phi"]

        if top_n is not None and top_n > 0:
            feature_contributions = feature_contributions[:top_n]

        # Predicted value from regressor
        pred_val = float(self.regressor_.predict(X_trans)[0])

        return {
            "prediction": round(pred_val, 2),
            "base_value": round(self.base_value_, 2),
            "features": feature_contributions,
        }

    def explain_global(
        self,
        snapshot_df: pd.DataFrame,
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """Calculate global mean absolute SHAP importance across encounters.

        Args:
            snapshot_df: DataFrame of patient snapshots.
            top_n: Maximum number of top features to return.

        Returns:
            List of dictionaries with feature name, mean_abs_shap, and rank.
        """
        if snapshot_df.empty:
            raise ValueError("Cannot compute global explanation on empty DataFrame.")

        # Guard against prohibited leakage
        leaked = set(snapshot_df.columns).intersection(PROHIBITED_FEATURE_COLUMNS)
        if leaked:
            raise ValueError(
                "Data leakage detected! Input contains prohibited columns: " f"{leaked}"
            )

        X_trans = self.preprocessor_.transform(snapshot_df)
        raw_shap = self.explainer_.shap_values(X_trans)
        if hasattr(raw_shap, "values"):
            raw_shap = raw_shap.values  # type: ignore[union-attr]

        raw_shap_2d = np.asarray(raw_shap)
        if raw_shap_2d.ndim == 1:
            raw_shap_2d = raw_shap_2d.reshape(1, -1)

        results: list[dict[str, Any]] = []
        for orig_col, indices in self.orig_to_transformed_indices_.items():
            if not indices:
                continue
            # Sum columns corresponding to original feature for each row
            col_shap_per_row = np.sum(raw_shap_2d[:, indices], axis=1)
            mean_abs = float(np.mean(np.abs(col_shap_per_row)))
            results.append(
                {
                    "name": orig_col,
                    "mean_abs_shap": round(mean_abs, 3),
                }
            )

        results.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank

        return results[:top_n]
