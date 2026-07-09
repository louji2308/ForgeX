from __future__ import annotations

import logging
from typing import Any

from forgex.errors import FairnessGateFailure, ForgeXError
from forgex.fairness.audit import FairnessReport, fairness_gate_for_promotion
from forgex.logging_setup import get_logger

logger = get_logger(__name__)

try:
    import mlflow
    from mlflow.exceptions import MlflowException
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    MlflowException = Exception


def evaluate_challenger(
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float],
    fairness_audit: FairnessReport | dict,
    min_relative_improvement: float = 0.02,
) -> dict[str, Any]:
    """A new model is only promoted if it beats the champion AND passes
    the fairness gate — performance alone doesn't clear the bar."""
    if isinstance(fairness_audit, dict):
        audit_obj = FairnessReport(
            demographic_parity_difference=fairness_audit.get("demographic_parity_difference", 0.0),
            equalized_odds_difference=fairness_audit.get("equalized_odds_difference", 0.0),
            selection_rate_by_group=fairness_audit.get("selection_rate_by_group", {}),
            passed_gate=fairness_audit.get("passed_gate", False),
            threshold=fairness_audit.get("threshold", 0.10),
        )
    else:
        audit_obj = fairness_audit

    challenger_pr_auc = challenger_metrics.get("pr_auc", 0.0)
    champion_pr_auc = champion_metrics.get("pr_auc", 0.0)

    pr_auc_gain = challenger_pr_auc - champion_pr_auc
    relative_gain = pr_auc_gain / max(champion_pr_auc, 1e-6)

    decision = {
        "pr_auc_gain": pr_auc_gain,
        "relative_gain": relative_gain,
        "fairness_passed": audit_obj.passed_gate,
        "promote": False,
        "reason": None,
    }

    try:
        fairness_gate_for_promotion(audit_obj)
    except FairnessGateFailure as e:
        decision["reason"] = f"blocked_by_fairness_gate: {e}"
        logger.warning(
            f"Challenger rejected: fairness gate failed "
            f"(DP diff={audit_obj.demographic_parity_difference:.3f})"
        )
        return decision

    if relative_gain < min_relative_improvement:
        decision["reason"] = (
            f"insufficient_improvement: {relative_gain:.2%} < required "
            f"{min_relative_improvement:.2%}"
        )
        logger.warning(f"Challenger rejected: {decision['reason']}")
        return decision

    decision["promote"] = True
    decision["reason"] = "passed_all_gates"
    logger.info(
        f"Challenger approved for promotion: PR-AUC gain={pr_auc_gain:.3f} "
        f"({relative_gain:.2%}), fairness passed"
    )
    return decision


def promote_to_champion(
    challenger_run_id: str,
    decision: dict[str, Any],
    mlflow_client=None,
    model_name: str = "forgex_hazard_model",
) -> None:
    if not decision.get("promote", False):
        raise RuntimeError(
            f"Refusing to promote: {decision.get('reason', 'unknown')}"
        )

    if not _MLFLOW_AVAILABLE:
        logger.warning(
            "MLflow not available — promotion logged but not registered."
        )
        return

    client = mlflow_client or mlflow.MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=challenger_run_id,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            f"Promoted run {challenger_run_id} to Production. "
            f"Reason: {decision['reason']}"
        )
    except MlflowException as e:
        raise RuntimeError(
            f"MLflow promotion failed for run {challenger_run_id}: {e}"
        ) from e
