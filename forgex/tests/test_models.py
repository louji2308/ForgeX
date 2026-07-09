import numpy as np
import pandas as pd
import pytest

from forgex.models.baseline import hazard_to_survival, check_coefficient_signs, EXPECTED_SIGNS
from forgex.models.hazard import ModelArtifact
from forgex.models.uplift import check_positivity, validate_cate_recovers_segments
from forgex.errors import ModelSchemaError, DataValidationError


def test_hazard_to_survival_valid():
    hazards = pd.Series([0.1, 0.2, 0.05])
    tenant_id = pd.Series(["T1", "T1", "T1"])
    month = pd.Series([1, 2, 3])

    result = hazard_to_survival(hazards, tenant_id, month)
    assert "survival_prob" in result.columns
    assert "cum_churn_prob" in result.columns
    assert all(result["survival_prob"] > 0)
    assert result["survival_prob"].iloc[0] == pytest.approx(0.9)
    assert result["survival_prob"].iloc[-1] < result["survival_prob"].iloc[0]


def test_hazard_to_survival_out_of_bounds():
    hazards = pd.Series([0.1, 1.5, 0.05])
    tenant_id = pd.Series(["T1", "T1", "T1"])
    month = pd.Series([1, 2, 3])

    with pytest.raises(ModelSchemaError, match="hazards must be in"):
        hazard_to_survival(hazards, tenant_id, month)


def test_hazard_to_survival_non_monotonic():
    hazards = pd.Series([0.5, 0.2, 0.7])
    tenant_id = pd.Series(["T1", "T1", "T1"])
    month = pd.Series([1, 2, 3])

    result = hazard_to_survival(hazards, tenant_id, month)
    cumprod = (1 - pd.Series([0.5, 0.2, 0.7])).cumprod()
    assert result["survival_prob"].iloc[0] == cumprod.iloc[0]


def test_check_positivity():
    df = pd.DataFrame({
        "tenant_id": [f"T{i}" for i in range(100)],
        "intervention_type": (["a"] * 40 + ["b"] * 30 + ["none"] * 30),
    })
    check_positivity(df, min_group_size=25)
    df2 = pd.DataFrame({
        "tenant_id": [f"T{i}" for i in range(10)],
        "intervention_type": (["a"] * 5 + ["none"] * 5),
    })
    with pytest.raises(DataValidationError):
        check_positivity(df2, min_group_size=6)


def test_check_positivity_no_control():
    df = pd.DataFrame({
        "tenant_id": [f"T{i}" for i in range(100)],
        "intervention_type": ["a"] * 50 + ["b"] * 50,
    })
    with pytest.raises(DataValidationError, match="No control"):
        check_positivity(df, min_group_size=25)


def test_model_artifact_creation():
    import lightgbm as lgb
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y = pd.Series([0, 1])
    ds = lgb.Dataset(X, y)
    booster = lgb.train({"objective": "binary", "verbosity": -1}, ds, num_boost_round=5)

    artifact = ModelArtifact(
        model=booster,
        feature_names=["f1", "f2"],
        categorical_features=[],
        train_event_rate=0.5,
        pr_auc=0.85,
    )
    assert artifact.pr_auc == 0.85
    assert len(artifact.feature_names) == 2
    assert artifact.trained_at is not None


def test_check_signs_no_violations():
    class MockModel:
        coef_ = np.array([[0.5, -0.3, 0.2]])

    check_coefficient_signs(
        MockModel(),
        ["days_late_trend_60d", "tenure_months", "complaint_severity_weighted_30d"],
    )
