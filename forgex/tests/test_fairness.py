import numpy as np
import pandas as pd
import pytest

from forgex.fairness.audit import (
    run_fairness_audit,
    fairness_gate_for_promotion,
    demonstrate_bias_correction,
    FairnessReport,
)
from forgex.errors import DataValidationError, FairnessGateFailure


def test_run_fairness_audit_single_group_raises():
    with pytest.raises(DataValidationError, match="only 1 unique value"):
        run_fairness_audit(
            y_true=pd.Series([0, 1, 0]),
            y_pred_binary=pd.Series([1, 1, 0]),
            sensitive_feature=pd.Series(["A", "A", "A"]),
        )


def test_run_fairness_audit_na_raises():
    with pytest.raises(DataValidationError, match="contains NaNs"):
        run_fairness_audit(
            y_true=pd.Series([0, 1, np.nan]),
            y_pred_binary=pd.Series([1, 1, 0]),
            sensitive_feature=pd.Series(["A", "B", "A"]),
        )


def test_run_fairness_audit_fair():
    result = run_fairness_audit(
        y_true=pd.Series([0, 1, 0, 1]),
        y_pred_binary=pd.Series([0, 1, 0, 1]),
        sensitive_feature=pd.Series(["A", "A", "B", "B"]),
        max_allowed_dp_diff=0.1,
    )
    assert result.passed_gate
    assert result.demographic_parity_difference == 0.0


def test_run_fairness_audit_unfair():
    result = run_fairness_audit(
        y_true=pd.Series([0, 1, 0, 1, 0, 0]),
        y_pred_binary=pd.Series([0, 1, 1, 1, 1, 1]),
        sensitive_feature=pd.Series(["A", "A", "A", "B", "B", "B"]),
        max_allowed_dp_diff=0.1,
    )
    # Group A: 1/3 positive, Group B: 3/3 positive -> DP diff = 0.67
    if not result.passed_gate:
        assert result.demographic_parity_difference > 0.1


def test_fairness_gate_for_promotion_passes():
    report = FairnessReport(demographic_parity_difference=0.05, passed_gate=True)
    fairness_gate_for_promotion(report)


def test_fairness_gate_for_promotion_fails():
    report = FairnessReport(demographic_parity_difference=0.5, passed_gate=False)
    with pytest.raises(FairnessGateFailure, match="Model failed fairness gate"):
        fairness_gate_for_promotion(report)


def test_demonstrate_bias_correction():
    result = demonstrate_bias_correction(
        historical_flag=pd.Series([1, 1, 1, 0, 0, 0]),
        new_model_pred=pd.Series([1, 1, 0, 0, 0, 0]),
        y_true=pd.Series([1, 1, 0, 0, 0, 1]),
        voucher_status=pd.Series([1, 1, 1, 0, 0, 0]),
    )
    assert "before" in result
    assert "after" in result
    assert "dp_improvement" in result
