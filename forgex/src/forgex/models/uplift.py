from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from forgex.errors import DataValidationError
from forgex.logging_setup import get_logger

logger = get_logger(__name__)


def check_positivity(
    intervention_log: pd.DataFrame,
    intervention_col: str = "intervention_type",
    min_group_size: int = 30,
) -> None:
    """The most-violated assumption in applied causal inference: if an arm
    was almost never given to some slice of tenants, any CATE estimate for
    that slice is extrapolation wearing inference's clothes. Refuse, don't
    warn."""
    counts = intervention_log.groupby(intervention_col).size()
    if len(thin := counts[counts < min_group_size]):
        raise DataValidationError(
            f"Intervention arms with < {min_group_size} historical examples: "
            f"{thin.to_dict()}. Generate more synthetic history for these "
            f"arms or drop them — do not fit CATE on them."
        )

    control_mask = intervention_log[intervention_col] == "none"
    if not control_mask.any() and "none" not in intervention_log[intervention_col].values:
        raise DataValidationError(
            "No control ('none') rows found — CATE needs a control arm."
        )


class TLearnerCATE:
    """T-learner: fits one model per treatment arm vs control.
    The interpretable default for causal uplift modeling."""

    def __init__(self, base_model=None):
        self.base_model = base_model or LogisticRegression(
            max_iter=2000, random_state=42, n_jobs=-1
        )
        self.models: dict[str, object] = {}
        self.arms: list[str] = []
        self.feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        Y: pd.Series,
        control_label: str = "none",
    ) -> "TLearnerCATE":
        self.feature_names = list(X.columns)
        self.arms = [a for a in T.unique() if a != control_label]

        control_mask = T == control_label
        X_control = X[control_mask]
        Y_control = Y[control_mask]

        for arm in self.arms:
            arm_mask = T == arm
            X_arm = pd.concat([X_control, X[arm_mask]], ignore_index=True)
            Y_arm = pd.concat([Y_control, Y[arm_mask]], ignore_index=True)

            if Y_arm.nunique() < 2:
                logger.warning(
                    f"Arm '{arm}' has degenerate outcome — skipping"
                )
                continue

            model = self.base_model.__class__(**self.base_model.get_params())
            model.fit(X_arm, Y_arm)
            self.models[arm] = model
            logger.info(f"T-learner fitted for arm '{arm}'")

        if not self.models:
            raise RuntimeError(
                "No CATE models were successfully fit — check intervention_log contents"
            )
        return self

    def predict_cate(self, X: pd.DataFrame) -> pd.DataFrame:
        results = []
        for arm, model in self.models.items():
            # Predict outcome if treated
            Y_t = model.predict_proba(X[self.feature_names])[:, 1]
            results.append(pd.DataFrame({
                "tenant_id": X.index if isinstance(X, pd.DataFrame) else range(len(X)),
                "arm": arm,
                "cate": Y_t,
            }))
        return pd.concat(results, ignore_index=True)


def fit_cate_models(
    intervention_log: pd.DataFrame,
    covariates: list[str],
    outcome_col: str = "renewed",
    intervention_col: str = "intervention_type",
    use_t_learner: bool = True,
) -> dict[str, object]:
    check_positivity(intervention_log, intervention_col)

    if use_t_learner:
        control_mask = intervention_log[intervention_col] == "none"
        learner = TLearnerCATE(
            base_model=GradientBoostingClassifier(
                max_depth=3, n_estimators=200, random_state=42
            )
        )
        X = intervention_log[covariates]
        T = intervention_log[intervention_col]
        Y = intervention_log[outcome_col]
        learner.fit(X, T, Y)
        return {"t_learner": learner}
    else:
        from econml.dml import CausalForestDML
        models: dict[str, object] = {}
        control_mask = intervention_log[intervention_col] == "none"

        for arm in intervention_log.loc[~control_mask, intervention_col].unique():
            subset = intervention_log[
                (intervention_log[intervention_col] == arm) | control_mask
            ]
            X, T_ = subset[covariates], (subset[intervention_col] == arm).astype(int)
            Y = subset[outcome_col]

            if T_.nunique() < 2 or Y.nunique() < 2:
                logger.warning(f"Skipping arm '{arm}': degenerate treatment or outcome variance")
                continue

            try:
                cf = CausalForestDML(
                    model_t=GradientBoostingClassifier(max_depth=3, n_estimators=200),
                    model_y=GradientBoostingRegressor(max_depth=3, n_estimators=200),
                    discrete_treatment=True,
                    n_estimators=200,
                    random_state=42,
                )
                cf.fit(Y, T_, X=X)
                models[arm] = cf
                logger.info(f"Causal forest fitted for arm '{arm}'")
            except Exception as e:
                raise RuntimeError(f"CATE fit failed for arm '{arm}': {e}") from e

        if not models:
            raise RuntimeError(
                "No CATE models were successfully fit — check intervention_log contents"
            )
        return models


def validate_cate_recovers_segments(
    cate_estimates: pd.DataFrame,
    hidden_segments: pd.DataFrame,
) -> dict:
    """This function IS the 'we recovered a segmentation we never labeled'
    claim, made testable instead of asserted on a slide."""
    merged = cate_estimates.merge(
        hidden_segments, on="tenant_id", validate="one_to_one"
    )
    if merged.empty:
        raise DataValidationError(
            "No overlap between cate_estimates and hidden_segments tenant_ids"
        )

    maint_cate = merged.loc[
        merged["true_segment"] == "maintenance_sensitive", "cate"
    ]
    price_cate = merged.loc[
        merged["true_segment"] == "price_sensitive", "cate"
    ]

    if len(maint_cate) < 5 or len(price_cate) < 5:
        logger.warning(
            f"Small segment sizes: maintenance_sensitive={len(maint_cate)}, "
            f"price_sensitive={len(price_cate)} — t-test may be unreliable"
        )

    t_stat, p_value = ttest_ind(maint_cate, price_cate, equal_var=False)

    return {
        "maintenance_sensitive_mean_cate": float(maint_cate.mean()),
        "price_sensitive_mean_cate": float(price_cate.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "recovers_segmentation": bool(
            p_value < 0.01 and maint_cate.mean() > price_cate.mean()
        ),
    }


def save_cate_models(
    models: dict[str, object], output_dir: str | Path
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = output_path / f"cate_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
    logger.info(f"Saved {len(models)} CATE models to {output_path}")


def load_cate_models(
    input_dir: str | Path,
) -> dict:
    input_path = Path(input_dir)
    models: dict = {}
    for pkl_path in input_path.glob("cate_*.pkl"):
        name = pkl_path.stem.replace("cate_", "")
        with open(pkl_path, "rb") as f:
            models[name] = pickle.load(f)
    return models
