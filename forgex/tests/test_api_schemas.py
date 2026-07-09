import pytest
from pydantic import ValidationError

from forgex.api.schemas import (
    ScoreRequest,
    WhatIfRequest,
    WhatIfResponse,
    OptimizeRequest,
    HealthResponse,
)


def test_score_request_valid():
    req = ScoreRequest(tenant_id="T000001")
    assert req.tenant_id == "T000001"


def test_what_if_request_valid():
    req = WhatIfRequest(
        tenant_id="T000001",
        rent_increase_pct=5.0,
        maintenance_speed="priority",
        retention_credit_usd=200.0,
    )
    assert req.rent_increase_pct == 5.0


def test_what_if_request_invalid_rent():
    with pytest.raises(ValidationError):
        WhatIfRequest(tenant_id="T1", rent_increase_pct=999.0)


def test_what_if_request_invalid_maintenance():
    with pytest.raises(ValidationError):
        WhatIfRequest(tenant_id="T1", maintenance_speed="turbo")


def test_what_if_request_invalid_credit():
    with pytest.raises(ValidationError):
        WhatIfRequest(tenant_id="T1", retention_credit_usd=99999.0)


def test_optimize_request_valid():
    req = OptimizeRequest(monthly_budget=10000.0, monthly_crew_hours=160.0)
    assert req.monthly_budget == 10000.0


def test_optimize_request_invalid_budget():
    with pytest.raises(ValidationError):
        OptimizeRequest(monthly_budget=-100.0)


def test_health_response_defaults():
    resp = HealthResponse()
    assert resp.status == "ok"
    assert resp.is_model_loaded is False


def test_what_if_response():
    resp = WhatIfResponse(
        baseline_risk_pct=45.0,
        scenario_risk_pct=32.0,
        delta_pts=-13.0,
        recommendation="Test recommendation",
    )
    assert resp.delta_pts == -13.0
