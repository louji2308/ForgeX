from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)

from forgex.errors import DataValidationError
from forgex.logging_setup import get_logger

logger = get_logger(__name__)

try:
    from lifelines.utils import concordance_index
    _LIFELINES_AVAILABLE = True
except ImportError:
    _LIFELINES_AVAILABLE = False


def compute_expected_retained_revenue(
    tenants: pd.DataFrame,
    allocation: pd.DataFrame,
    baseline_policy: str = "no_intervention",
) -> dict:
    if baseline_policy not in {"no_intervention", "random"}:
        raise ValueError(f"Unknown baseline_policy: {baseline_policy}")

    treated = tenants[tenants["tenant_id"].isin(set(allocation["tenant_id"]))]
    if treated.empty:
        raise ValueError("allocation selected zero tenants — nothing to evaluate")

    optimized_value = (treated["cate"] * treated["ltv"]).sum()

    if baseline_policy == "no_intervention":
        baseline_value = 0.0
    else:
        random_sample = tenants.sample(
            n=min(len(treated), len(tenants)), random_state=0
        )
        baseline_value = (random_sample["cate"] * random_sample["ltv"]).sum()

    if optimized_value == 0:
        logger.warning(
            "Optimized policy value is exactly 0 — check CATE/LTV inputs "
            "before reporting this"
        )

    return {
        "optimized_expected_value_usd": float(optimized_value),
        "baseline_policy": baseline_policy,
        "baseline_value_usd": float(baseline_value),
        "lift_usd": float(optimized_value - baseline_value),
        "n_tenants_treated": int(len(treated)),
    }


def compute_evaluation_metrics(
    y_true: pd.Series,
    y_pred_prob: pd.Series,
    y_pred_binary: pd.Series | None = None,
    tenant_ids: pd.Series | None = None,
    event_times: pd.Series | None = None,
) -> dict:
    if y_pred_binary is None:
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)

    event_rate = y_true.mean()

    pr_auc = average_precision_score(y_true, y_pred_prob)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    brier = brier_score_loss(y_true, y_pred_prob)
    f2 = fbeta_score(y_true, y_pred_binary, beta=2)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    max_f1 = float(f1_scores.max())

    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    calibration_error = float(np.abs(prob_true - prob_pred).mean())

    c_index = None
    if _LIFELINES_AVAILABLE and tenant_ids is not None and event_times is not None:
        try:
            c_index = float(
                concordance_index(event_times, -y_pred_prob, event_observed=y_true)
            )
        except Exception as e:
            logger.warning(f"Could not compute C-index: {e}")

    return {
        "event_rate": float(event_rate),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
        "f2_score": float(f2),
        "max_f1": max_f1,
        "calibration_error": calibration_error,
        "c_index": c_index,
        "n_samples": int(len(y_true)),
        "n_events": int(y_true.sum()),
    }


def generate_evaluation_report(
    person_period: pd.DataFrame,
    model_artifact,
    feature_names: list[str],
    output_path: str | Path | None = None,
) -> dict:
    """Generate a reproducible evaluation report across all held-out data."""
    test = person_period[person_period["fold"] == "test"].copy()
    if test.empty:
        test = person_period[person_period["fold"] == "val"].copy()
    if test.empty:
        test = person_period.copy()
        logger.warning("No test/val fold found — evaluating on all data")

    X_test = test[feature_names].fillna(0)
    y_test = test["churn_event_this_month"]
    tenant_ids_test = test["tenant_id"]

    y_pred_prob = model_artifact.model.predict(X_test)
    if hasattr(model_artifact.model, "_calibrator"):
        y_pred_prob = model_artifact.model._calibrator.predict(y_pred_prob)

    y_pred_prob = pd.Series(y_pred_prob, index=y_test.index)
    y_pred_binary = (y_pred_prob >= model_artifact.train_event_rate).astype(int)

    metrics = compute_evaluation_metrics(
        y_test, y_pred_prob, y_pred_binary,
        tenant_ids=tenant_ids_test,
        event_times=test["month_of_lease"],
    )

    report = {
        "model_info": {
            "trained_at": model_artifact.trained_at,
            "feature_count": len(model_artifact.feature_names),
            "train_event_rate": model_artifact.train_event_rate,
            "calibration_applied": getattr(model_artifact, "calibration_applied", False),
        },
        "test_metrics": metrics,
        "threshold": float(model_artifact.train_event_rate),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df = pd.DataFrame([report])
        report_df.to_json(output_path, orient="records", indent=2)
        logger.info(f"Evaluation report saved to {output_path}")

    return report
