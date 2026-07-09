from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from forgex.errors import DataValidationError, ModelSchemaError
from forgex.logging_setup import get_logger

logger = get_logger(__name__)


EXPECTED_SIGNS = {
    "days_late_trend_60d": 1,
    "complaint_severity_weighted_30d": 1,
    "tenure_months": -1,
    "rent_gap_pct": 1,
    "market_rent_gap": 1,
    "household_size": 0,
    "is_voucher_holder": 0,
}


def hazard_to_survival(
    hazards: pd.Series,
    tenant_id: pd.Series,
    month: pd.Series,
) -> pd.DataFrame:
    """S(t) = Π_{k<=t}(1 - h_k). Every downstream 'P(churn by month k)'
    claim depends on this function, so it gets unit tests, not an inline
    .cumprod() someone half-trusts."""
    if not ((hazards >= 0) & (hazards <= 1)).all():
        bad = hazards[(hazards < 0) | (hazards > 1)]
        raise ModelSchemaError(
            f"hazards must be in [0,1], found {len(bad)} out-of-range values"
        )

    df = pd.DataFrame({"tenant_id": tenant_id, "month": month, "hazard": hazards})
    df = df.sort_values(["tenant_id", "month"])
    df["survival_prob"] = df.groupby("tenant_id")["hazard"].transform(
        lambda h: (1 - h).cumprod()
    )
    df["cum_churn_prob"] = 1 - df["survival_prob"]

    non_monotone = df.groupby("tenant_id")["survival_prob"].apply(
        lambda s: (s.diff().dropna() > 1e-9).any()
    )
    if non_monotone.any():
        offenders = non_monotone[non_monotone].index.tolist()[:5]
        raise ModelSchemaError(
            f"Survival curve increased for tenants {offenders} — cumprod logic broken"
        )
    return df


def check_coefficient_signs(model, feature_names: list[str]) -> None:
    coefs = dict(zip(feature_names, model.coef_.ravel()))
    violations = [
        f"{f}: expected sign {EXPECTED_SIGNS[f]}, got {coefs[f]:.4f}"
        for f in EXPECTED_SIGNS
        if f in coefs and (1 if coefs[f] > 0 else -1 if coefs[f] < 0 else 0) != EXPECTED_SIGNS[f]
    ]
    if violations:
        warnings.warn(
            "Coefficient sign check failed:\n  " + "\n  ".join(violations),
            stacklevel=2,
        )


@dataclass
class BaselineArtifact:
    model: LogisticRegression
    feature_names: list[str]
    train_event_rate: float
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pr_auc: float = 0.0


class BaselineHazardModel:
    """A small, fully interpretable model that must always work.
    If every other phase catches fire before the demo, this is what
    you fall back to."""

    def __init__(self, feature_subset: list[str] | None = None):
        self.feature_subset = feature_subset
        self.artifact: BaselineArtifact | None = None

    def _select_features(self, X: pd.DataFrame) -> list[str]:
        if self.feature_subset:
            missing = [f for f in self.feature_subset if f not in X.columns]
            if missing:
                raise DataValidationError(
                    f"Requested features not in training data: {missing}"
                )
            return self.feature_subset

        forbidden = {
            "tenant_id", "lease_id", "calendar_month", "month_of_lease",
            "churn_event_this_month", "is_censored", "still_active",
            "fold", "as_of_month",
        }
        candidates = [c for c in X.columns if c not in forbidden]
        numeric = X[candidates].select_dtypes(include=[np.number]).columns.tolist()
        return numeric

    def fit(
        self,
        person_period: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> BaselineArtifact:
        train = person_period[person_period["fold"] == "train"].copy()
        if train.empty:
            raise DataValidationError("No training rows found in person_period")

        feature_cols = feature_cols or self._select_features(train)
        missing = [f for f in feature_cols if f not in train.columns]
        if missing:
            raise DataValidationError(f"Features missing from training data: {missing}")

        # Keep only numeric columns for model training
        feature_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = train[feature_cols].fillna(0)
        y = train["churn_event_this_month"]

        if y.nunique() < 2:
            raise DataValidationError(
                f"y has only {y.nunique()} class(es) — check event distribution"
            )

        event_rate = y.mean()
        scale_pos_weight = (1 - event_rate) / max(event_rate, 1e-6)

        model = LogisticRegression(
            class_weight={0: 1.0, 1: scale_pos_weight},
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)

        train_pred = model.predict_proba(X)[:, 1]
        pr_auc = average_precision_score(y, train_pred)

        check_coefficient_signs(model, feature_cols)

        self.artifact = BaselineArtifact(
            model=model,
            feature_names=feature_cols,
            train_event_rate=float(event_rate),
            pr_auc=float(pr_auc),
        )

        logger.info(
            f"Baseline model trained: PR-AUC={pr_auc:.3f}, "
            f"event_rate={event_rate:.3f}"
        )
        return self.artifact

    def predict_hazards(self, person_period: pd.DataFrame) -> np.ndarray:
        if self.artifact is None:
            raise RuntimeError("Model not fitted yet — call .fit() first")

        missing = [f for f in self.artifact.feature_names if f not in person_period.columns]
        if missing:
            raise ModelSchemaError(
                f"Features missing at prediction time: {missing}"
            )

        X = person_period[self.artifact.feature_names].fillna(0)
        return self.artifact.model.predict_proba(X)[:, 1]

    def predict_survival_curves(
        self, person_period: pd.DataFrame,
    ) -> pd.DataFrame:
        hazards = self.predict_hazards(person_period)
        return hazard_to_survival(
            pd.Series(hazards),
            person_period["tenant_id"],
            person_period["month_of_lease"],
        )
