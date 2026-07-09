"""Shared test fixtures for the ForgeX test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def sample_leases() -> pd.DataFrame:
    return pd.DataFrame({
        "tenant_id": ["T1", "T2"],
        "lease_id": ["L1", "L2"],
        "lease_start": [pd.Timestamp("2024-01-01"), pd.Timestamp("2023-06-01")],
        "lease_end": [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-01")],
        "did_renew": [False, True],
    })


@pytest.fixture
def sample_tenants() -> pd.DataFrame:
    return pd.DataFrame({
        "tenant_id": ["T1", "T2"],
        "household_size": [2, 4],
        "voucher_holder": [False, True],
        "zip_code": ["ZIP_001", "ZIP_002"],
        "move_in_date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2022-06-01")],
        "hidden_segment": ["price_sensitive", "maintenance_sensitive"],
        "_satisfaction_base": [0.7, 0.5],
        "_financial_stress_base": [0.2, 0.4],
    })


@pytest.fixture
def sample_person_period() -> pd.DataFrame:
    data = []
    for tid, n_months, churn_month in [("T1", 6, 6), ("T2", 12, None)]:
        for m in range(1, n_months + 1):
            is_churn = 1 if churn_month and m == churn_month else 0
            data.append({
                "tenant_id": tid,
                "lease_id": f"L{tid[-1]}",
                "month_of_lease": m,
                "calendar_month": pd.Timestamp(f"2024-{m:02d}-01"),
                "churn_event_this_month": is_churn,
                "is_censored": 0,
                "still_active": 1 if m < n_months else 0,
                "fold": "train",
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_feature_table() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2024-06-01", freq="MS")
    rows = []
    for tid in ["T1", "T2"]:
        for d in dates:
            rows.append({
                "tenant_id": tid,
                "as_of_month": d,
                "days_late_trend_60d": 0.5,
                "complaint_severity_weighted_30d": 2.0,
                "tenure_months": 12,
                "rent_gap_pct": 0.05,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_intervention_log() -> pd.DataFrame:
    return pd.DataFrame({
        "tenant_id": ["T1", "T1", "T2", "T2"] * 25,
        "intervention_type": (["retention_credit"] * 25 + ["none"] * 25 +
                              ["priority_maintenance"] * 25 + ["none"] * 25),
        "renewed": [True] * 50 + [False] * 25 + [True] * 25,
        "cost_dollars": [150.0] * 25 + [0.0] * 25 + [75.0] * 25 + [0.0] * 25,
        "crew_hours": [1.0] * 25 + [0.0] * 25 + [3.0] * 25 + [0.0] * 25,
    })
