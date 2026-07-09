from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from forgex.errors import ModelSchemaError
from forgex.logging_setup import get_logger

logger = get_logger(__name__)

DRIVER_LABELS = {
    "days_late_trend_60d": "recent late payments",
    "complaint_severity_weighted_30d": "unresolved maintenance issues",
    "complaint_count_30d": "recent complaints",
    "complaint_count_60d": "recent complaints",
    "complaint_severity_weighted_60d": "unresolved maintenance issues",
    "complaint_severity_weighted_90d": "unresolved maintenance issues",
    "days_late_avg_30d": "average payment lateness",
    "days_late_avg_60d": "average payment lateness",
    "days_late_avg_90d": "average payment lateness",
    "late_payment_count_30d": "recent late payments",
    "late_payment_count_60d": "recent late payments",
    "tenure_months": "tenure discount",
    "rent_gap_pct": "rent increase burden",
    "market_rent_gap": "below-market rent advantage",
    "household_size": "household size",
    "is_voucher_holder": "voucher holder status",
    "household_size_large": "large household",
    "complaint_sentiment_avg_30d": "complaint sentiment",
    "avg_resolve_days_30d": "maintenance resolution speed",
    "prior_interventions_180d": "prior retention offers",
    "prior_intervention_accept_rate_180d": "prior offer acceptance",
    "lease_remaining_months": "lease term remaining",
    "payment_count_30d": "payment consistency",
    "payment_count_60d": "payment consistency",
    "payment_count_90d": "payment consistency",
    "total_late_amount_30d": "total late amount",
    "total_late_amount_60d": "total late amount",
    "total_late_amount_90d": "total late amount",
    "late_payment_count_90d": "recent late payments",
    "days_late_trend_30d": "recent late payment trend",
    "days_late_trend_90d": "late payment trend",
    "complaint_sentiment_avg_60d": "complaint sentiment",
    "complaint_sentiment_avg_90d": "complaint sentiment",
    "avg_resolve_days_60d": "maintenance resolution speed",
    "avg_resolve_days_90d": "maintenance resolution speed",
    "complaint_count_90d": "recent complaints",
    "current_rent": "current monthly rent",
    "avg_market_rent": "average market rent",
    "prior_interventions_365d": "prior retention offers",
    "prior_intervention_accept_rate_365d": "prior offer acceptance",
}


def top_shap_drivers(
    shap_row: pd.Series,
    feature_row: pd.Series,
    k: int = 3,
) -> list[dict]:
    if unmapped := set(shap_row.index) - set(DRIVER_LABELS.keys()):
        raise ModelSchemaError(
            f"{len(unmapped)} SHAP features have no human-readable label: "
            f"{sorted(unmapped)[:5]}. Add to DRIVER_LABELS before shipping."
        )
    ranked = shap_row.abs().sort_values(ascending=False).head(k)
    return [
        {
            "feature": feat,
            "label": DRIVER_LABELS[feat],
            "shap_value": float(shap_row[feat]),
            "direction": "increases_risk" if shap_row[feat] > 0 else "decreases_risk",
            "raw_value": feature_row.get(feat),
        }
        for feat in ranked.index
    ]


class ShapExplainer:
    """Computes SHAP values using appropriate explainer for model type."""

    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._background: pd.DataFrame | None = None
        self._is_tree_model = self._detect_tree_model(model)

    def _detect_tree_model(self, model) -> bool:
        """Check if model is tree-based (LightGBM, XGBoost, sklearn tree)."""
        model_type = type(model).__name__
        tree_types = {
            "LGBMModel", "LGBMClassifier", "LGBMRegressor",
            "XGBClassifier", "XGBRegressor", "XGBModel",
            "DecisionTreeClassifier", "DecisionTreeRegressor",
            "RandomForestClassifier", "RandomForestRegressor",
            "GradientBoostingClassifier", "GradientBoostingRegressor",
            "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
        }
        return model_type in tree_types

    def fit(self, background_data: pd.DataFrame, sample_size: int = 500) -> "ShapExplainer":
        if len(background_data) > sample_size:
            background_data = background_data.sample(sample_size, random_state=42)

        self._background = background_data[self.feature_names].fillna(0)
        
        if self._is_tree_model:
            self._explainer = shap.TreeExplainer(self.model, data=self._background)
        else:
            # Use LinearExplainer for linear models (LogisticRegression, LinearRegression)
            self._explainer = shap.LinearExplainer(self.model, self._background, feature_perturbation="interventional")
        
        logger.info(
            f"SHAP explainer initialized with {len(self._background)} background samples "
            f"({'tree' if self._is_tree_model else 'linear'} model)"
        )
        return self

    def explain(self, features: pd.DataFrame) -> np.ndarray:
        if self._explainer is None:
            raise RuntimeError("Explainer not fitted — call .fit() first")

        missing = [f for f in self.feature_names if f not in features.columns]
        if missing:
            raise ModelSchemaError(
                f"Features missing at explanation time: {missing}"
            )

        X = features[self.feature_names].fillna(0)
        shap_values = self._explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return np.asarray(shap_values)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"SHAP explainer saved to {path}")

    @staticmethod
    def load(path: str | Path) -> "ShapExplainer":
        path = Path(path)
        if not path.exists():
            raise ModelSchemaError(f"SHAP explainer not found at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
