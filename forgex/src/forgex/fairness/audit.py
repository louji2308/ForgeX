from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forgex.errors import DataValidationError, FairnessGateFailure
from forgex.logging_setup import get_logger

logger = get_logger(__name__)

try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        equalized_odds_difference,
    )
    _FAIRLEARN_AVAILABLE = True
except ImportError:
    _FAIRLEARN_AVAILABLE = False
    logger.warning("fairlearn not available — using simplified fairness metrics")


@dataclass
class FairnessReport:
    demographic_parity_difference: float = 0.0
    equalized_odds_difference: float = 0.0
    selection_rate_by_group: dict[str, float] = field(default_factory=dict)
    passed_gate: bool = True
    threshold: float = 0.1
    n_groups: int = 0
    group_sizes: dict[str, int] = field(default_factory=dict)


def _simple_demographic_parity(
    y_pred_binary: pd.Series,
    sensitive_feature: pd.Series,
) -> float:
    """Simplified demographic parity calculation when fairlearn is unavailable."""
    rates = y_pred_binary.groupby(sensitive_feature).mean()
    if len(rates) < 2:
        return 0.0
    return float(rates.max() - rates.min())


def run_fairness_audit(
    y_true: pd.Series,
    y_pred_binary: pd.Series,
    sensitive_feature: pd.Series,
    max_allowed_dp_diff: float = 0.10,
) -> FairnessReport:
    if sensitive_feature.nunique() < 2:
        raise DataValidationError(
            f"sensitive_feature has only {sensitive_feature.nunique()} unique value(s) — "
            f"cannot compute a between-group disparity metric."
        )

    for name, series in [
        ("y_true", y_true),
        ("y_pred_binary", y_pred_binary),
        ("sensitive_feature", sensitive_feature),
    ]:
        if series.isna().any():
            raise DataValidationError(
                f"{name} contains NaNs — fairness metrics will silently misbehave"
            )

    group_sizes = sensitive_feature.value_counts().to_dict()

    if _FAIRLEARN_AVAILABLE:
        dp_diff = demographic_parity_difference(
            y_true, y_pred_binary, sensitive_features=sensitive_feature
        )
        eo_diff = equalized_odds_difference(
            y_true, y_pred_binary, sensitive_features=sensitive_feature
        )
        frame = MetricFrame(
            metrics={"selection_rate": lambda yt, yp: float(yp.mean())},
            y_true=y_true,
            y_pred=y_pred_binary,
            sensitive_features=sensitive_feature,
        )
        selection_rates = {
            str(k): float(v)
            for k, v in frame.by_group.to_dict()["selection_rate"].items()
        }
    else:
        dp_diff = _simple_demographic_parity(y_pred_binary, sensitive_feature)
        eo_diff = 0.0
        selection_rates = {
            str(k): float(v)
            for k, v in y_pred_binary.groupby(sensitive_feature).mean().items()
        }

    return FairnessReport(
        demographic_parity_difference=float(dp_diff),
        equalized_odds_difference=float(eo_diff),
        selection_rate_by_group=selection_rates,
        passed_gate=bool(abs(dp_diff) <= max_allowed_dp_diff),
        threshold=max_allowed_dp_diff,
        n_groups=sensitive_feature.nunique(),
        group_sizes=group_sizes,
    )


def fairness_gate_for_promotion(audit_result: FairnessReport) -> None:
    """Called from Phase 15's promotion step. Raising here — instead of
    returning False — is deliberate: a model that fails fairness must be
    structurally unable to reach production, not flagged for someone to
    notice later."""
    if not audit_result.passed_gate:
        raise FairnessGateFailure(
            f"Model failed fairness gate: demographic_parity_difference="
            f"{audit_result.demographic_parity_difference:.3f} exceeds "
            f"threshold {audit_result.threshold}. Promotion blocked."
        )
    logger.info("Fairness gate passed — model cleared for promotion.")


def demonstrate_bias_correction(
    historical_flag: pd.Series,
    new_model_pred: pd.Series,
    y_true: pd.Series,
    voucher_status: pd.Series,
    max_allowed_dp_diff: float = 0.10,
) -> dict[str, Any]:
    """Demonstrates the reduction in demographic disparity between the
    legacy biased flag and the new model's predictions.

    This is the provable 'before vs. after' story for the fairness demo.
    """
    before = run_fairness_audit(
        y_true, historical_flag, voucher_status, max_allowed_dp_diff
    )
    after = run_fairness_audit(
        y_true, new_model_pred, voucher_status, max_allowed_dp_diff
    )

    improvement = abs(before.demographic_parity_difference) - abs(
        after.demographic_parity_difference
    )

    if improvement <= 0:
        logger.warning(
            f"New model did NOT improve fairness vs. the legacy flag "
            f"(before={before.demographic_parity_difference:.3f}, "
            f"after={after.demographic_parity_difference:.3f}). Do not present "
            f"this as a fixed-bias demo until it's actually true."
        )

    return {
        "before": {
            "demographic_parity_difference": before.demographic_parity_difference,
            "equalized_odds_difference": before.equalized_odds_difference,
            "selection_rate_by_group": before.selection_rate_by_group,
            "passed_gate": before.passed_gate,
        },
        "after": {
            "demographic_parity_difference": after.demographic_parity_difference,
            "equalized_odds_difference": after.equalized_odds_difference,
            "selection_rate_by_group": after.selection_rate_by_group,
            "passed_gate": after.passed_gate,
        },
        "dp_improvement": improvement,
        "fairness_fixed": improvement > 0 and after.passed_gate,
    }
