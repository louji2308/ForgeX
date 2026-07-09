from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    tenant_id: str


class DriverDetail(BaseModel):
    feature: str
    label: str
    shap_value: float
    direction: str
    raw_value: float | None = None


class ScoreResponse(BaseModel):
    tenant_id: str
    risk_pct: float
    risk_band: str
    survival_prob: float | None = None
    top_drivers: list[DriverDetail] = []
    narrative: str = ""
    narrative_source: str = ""


class WhatIfRequest(BaseModel):
    tenant_id: str
    rent_increase_pct: float = Field(default=0.0, ge=0, le=50)
    maintenance_speed: Literal["standard", "priority", "same_day"] = "standard"
    retention_credit_usd: float = Field(default=0.0, ge=0, le=2000)


class WhatIfResponse(BaseModel):
    baseline_risk_pct: float
    scenario_risk_pct: float
    delta_pts: float
    recommendation: str
    narrative: str = ""


class OptimizeRequest(BaseModel):
    monthly_budget: float = Field(default=5000.0, gt=0)
    monthly_crew_hours: float = Field(default=80.0, gt=0)


class OptimizeResponse(BaseModel):
    selections: list[dict]
    total_selected: int
    total_budget_spent: float
    solver_status: str


class HealthResponse(BaseModel):
    status: str = "ok"
    is_model_loaded: bool = False
    is_explainer_loaded: bool = False
    is_cate_loaded: bool = False
    tenants_indexed: int = 0
