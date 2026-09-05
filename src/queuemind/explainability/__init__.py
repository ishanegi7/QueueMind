"""QueueMind explainability package for patient-flow models.

Provides TreeSHAP-based local and global feature attribution utilities:
- ShapExplainer: Production TreeSHAP explainer with human-readable feature aggregation.
"""

from queuemind.explainability.shap_explainer import ShapExplainer

__all__ = ["ShapExplainer"]
