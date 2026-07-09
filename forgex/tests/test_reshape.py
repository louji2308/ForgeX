import pandas as pd
import pytest

from forgex.reshape.person_period import (
    explode_to_person_period,
    validate_person_period,
    tenant_level_split,
)
from forgex.errors import DataValidationError


def test_explode_to_person_period_basic():
    leases = pd.DataFrame({
        "tenant_id": ["T1"],
        "lease_id": ["L1"],
        "lease_start": [pd.Timestamp("2024-01-01")],
        "lease_end": [pd.Timestamp("2024-06-01")],
        "did_renew": [False],
    })
    result = explode_to_person_period(leases, as_of_date=pd.Timestamp("2024-06-01"))
    assert len(result) == 6
    assert result["churn_event_this_month"].sum() == 1
    assert result.iloc[-1]["churn_event_this_month"] == 1


def test_explode_to_person_period_renewed():
    leases = pd.DataFrame({
        "tenant_id": ["T1"],
        "lease_id": ["L1"],
        "lease_start": [pd.Timestamp("2024-01-01")],
        "lease_end": [pd.Timestamp("2024-03-01")],
        "did_renew": [True],
    })
    result = explode_to_person_period(leases, as_of_date=pd.Timestamp("2024-03-01"))
    assert result["churn_event_this_month"].sum() == 0
    assert result.iloc[-1]["is_censored"] == 0


def test_explode_to_person_period_censored():
    leases = pd.DataFrame({
        "tenant_id": ["T1"],
        "lease_id": ["L1"],
        "lease_start": [pd.Timestamp("2024-01-01")],
        "lease_end": [pd.Timestamp("2024-06-01")],
        "did_renew": [True],
    })
    result = explode_to_person_period(leases, as_of_date=pd.Timestamp("2024-03-01"))
    assert result["churn_event_this_month"].sum() == 0
    assert result.iloc[-1]["is_censored"] == 1


def test_explode_to_person_period_missing_columns():
    with pytest.raises(DataValidationError, match="leases missing required columns"):
        explode_to_person_period(pd.DataFrame({"a": [1]}))


def test_validate_person_period():
    pp = pd.DataFrame({
        "lease_id": ["L1", "L1"],
        "tenant_id": ["T1", "T1"],
        "month_of_lease": [1, 2],
        "churn_event_this_month": [0, 1],
        "is_censored": [0, 0],
        "still_active": [1, 0],
    })
    leases = pd.DataFrame({"lease_id": ["L1"]})
    validate_person_period(pp, leases)


def test_validate_person_period_missing_leases():
    pp = pd.DataFrame({
        "lease_id": ["L1"],
        "tenant_id": ["T1"],
        "month_of_lease": [1],
        "churn_event_this_month": [0],
        "is_censored": [0],
        "still_active": [1],
    })
    leases = pd.DataFrame({"lease_id": ["L1", "L2"]})
    with pytest.raises(DataValidationError, match="leases produced zero"):
        validate_person_period(pp, leases)


def test_tenant_level_split():
    tenant_ids = pd.Series([f"T{i}" for i in range(100)])
    train, val, test = tenant_level_split(tenant_ids, test_frac=0.15, val_frac=0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
