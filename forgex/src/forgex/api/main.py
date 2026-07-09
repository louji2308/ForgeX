from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from forgex.api.schemas import (
    AiSearchRequest,
    AiSearchResponse,
    HealthResponse,
    LeverDetail,
    OptimizeRequest,
    OptimizeResponse,
    ScoreResponse,
    WhatIfRequest,
    WhatIfResponse,
    DriverDetail,
)
from forgex.auth.firebase_config import get_db
from forgex.auth.routes import router as auth_router
from forgex.config import load_settings
from forgex.errors import (
    FairnessGateFailure,
    FeatureBuildError,
    ForgeXError,
    ModelSchemaError,
)
from forgex.explain.narrative import generate_narrative
from forgex.explain.shap_explainer import ShapExplainer, top_shap_drivers, DRIVER_LABELS
from forgex.logging_setup import get_logger
from forgex.models.baseline import hazard_to_survival
from forgex.models.hazard import ModelArtifact, load_model_artifact
from forgex.models.uplift import load_cate_models
from forgex.optimize.ilp_optimizer import solve_retention_allocation, solve_retention_allocation_greedy

logger = get_logger(__name__)
settings = load_settings()


class AppState:
    def __init__(self):
        self.hazard_model: ModelArtifact | None = None
        self.explainer: ShapExplainer | None = None
        self.cate_models: dict | None = None
        self.feature_table: pd.DataFrame | None = None
        self.person_period: pd.DataFrame | None = None
        self.tenant_index: set[str] = set()
        self.tenants_df: pd.DataFrame | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading artifacts at startup...")
        state.hazard_model = load_model_artifact(
            str(settings.artifacts_dir / "hazard_model.pkl"),
        )
        state.explainer = ShapExplainer.load(
            settings.artifacts_dir / "shap_explainer.pkl"
        )
        state.cate_models = load_cate_models(settings.artifacts_dir / "cate")

        pp_path = settings.data_dir / "processed" / "person_period.parquet"
        feat_path = settings.data_dir / "processed" / "feature_table.parquet"
        ten_path = settings.data_dir / "raw" / "tenants.parquet"

        if pp_path.exists():
            state.person_period = pd.read_parquet(pp_path)
        if feat_path.exists():
            state.feature_table = pd.read_parquet(feat_path)
        if ten_path.exists():
            state.tenants_df = pd.read_parquet(ten_path)

        if state.tenants_df is not None:
            state.tenant_index = set(state.tenants_df["tenant_id"].unique())

        logger.info(
            f"Startup complete: model={'ok' if state.hazard_model else 'missing'}, "
            f"explainer={'ok' if state.explainer else 'missing'}, "
            f"tenants={len(state.tenant_index)}"
        )
    except (FileNotFoundError, ModelSchemaError) as e:
        logger.critical(f"Startup failed — could not load artifacts: {e}")
        raise
    yield


app = FastAPI(
    title="ForgeX API",
    description="Vital-signs monitoring for the landlord-tenant relationship",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


@app.middleware("http")
async def cors_fallback(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault("Access-Control-Allow-Methods", "*")
    response.headers.setdefault("Access-Control-Allow-Headers", "*")
    if request.method == "OPTIONS":
        return Response(status_code=204, headers={
            "access-control-allow-origin": "*",
            "access-control-allow-methods": "*",
            "access-control-allow-headers": "*",
            "access-control-max-age": "86400",
        })
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    logger.error(
        f"[{request_id}] Unhandled error on {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "request_id": request_id,
            "message": str(exc),
        },
        headers={"Access-Control-Allow-Origin": "*"},
    )


def _synth_features_for_new_tenant(tenant_id: str) -> pd.DataFrame:
    """Build synthetic feature row for a tenant not yet in person_period."""
    if state.hazard_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = {}
    for col in state.hazard_model.feature_names:
        if state.person_period is not None and col in state.person_period.columns:
            row[col] = float(state.person_period[col].median())
        elif state.tenants_df is not None and col in state.tenants_df.columns:
            row[col] = float(state.tenants_df[col].median())
        else:
            row[col] = 0.0
    if "tenure_months" in row:
        row["tenure_months"] = 1.0
    return pd.DataFrame([row])


def _score_tenant(
    tenant_id: str,
) -> tuple[float, list[dict] | None, np.ndarray | None]:
    """Internal scoring function shared by /score and /simulate."""
    if state.hazard_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — train or restart with artifacts")

    if state.person_period is None:
        raise FeatureBuildError("person_period not loaded")

    tenant_rows = state.person_period[
        state.person_period["tenant_id"] == tenant_id
    ].sort_values("month_of_lease")

    if tenant_rows.empty:
        latest = _synth_features_for_new_tenant(tenant_id)
    else:
        latest = tenant_rows.iloc[-1:]

    if hasattr(state.hazard_model.model, "predict_proba"):
        hazard_probs = state.hazard_model.model.predict_proba(
            latest[state.hazard_model.feature_names].fillna(0)
        )[:, 1]
    else:
        hazard_probs = state.hazard_model.model.predict(
            latest[state.hazard_model.feature_names].fillna(0)
        )
    if hasattr(state.hazard_model.model, "_calibrator"):
        hazard_probs = state.hazard_model.model._calibrator.predict(hazard_probs)

    risk_pct = float(hazard_probs[0] * 100)

    drivers = None
    shap_values_raw = None
    if state.explainer is not None:
        shap_values_raw = state.explainer.explain(latest)
        shap_series = pd.Series(
            shap_values_raw[0], index=state.hazard_model.feature_names
        )
        feature_row = latest[state.hazard_model.feature_names].iloc[0]
        drivers = top_shap_drivers(shap_series, feature_row, k=3)

    return risk_pct, drivers, hazard_probs


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/tenants")
def list_tenants():
    all_tenants = set(state.tenant_index) if state.tenant_index else set()
    try:
        fb_docs = get_db().collection("tenant_profiles").stream()
        for d in fb_docs:
            all_tenants.add(d.id)
    except Exception:
        pass
    if all_tenants:
        return {"tenants": sorted(all_tenants)}
    return {"tenants": []}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        is_model_loaded=state.hazard_model is not None,
        is_explainer_loaded=state.explainer is not None,
        is_cate_loaded=state.cate_models is not None and len(state.cate_models) > 0,
        tenants_indexed=len(state.tenant_index),
    )


@app.get("/score/{tenant_id}", response_model=ScoreResponse)
def score(tenant_id: str):
    try:
        risk_pct, drivers, hazards = _score_tenant(tenant_id)
    except (FeatureBuildError, ModelSchemaError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    risk_band = "Critical" if risk_pct >= 80 else "High" if risk_pct >= 60 else "Moderate" if risk_pct >= 40 else "Low" if risk_pct >= 20 else "Very Low"

    survival_prob = None
    is_estimate = tenant_id not in state.tenant_index
    if hazards is not None and state.person_period is not None:
        tenant_rows = state.person_period[
            state.person_period["tenant_id"] == tenant_id
        ]
        if len(tenant_rows) > 0:
            if hasattr(state.hazard_model.model, "predict_proba"):
                all_hazards = state.hazard_model.model.predict_proba(
                    tenant_rows[state.hazard_model.feature_names].fillna(0)
                )[:, 1]
            else:
                all_hazards = state.hazard_model.model.predict(
                    tenant_rows[state.hazard_model.feature_names].fillna(0)
                )
            if hasattr(state.hazard_model.model, "_calibrator"):
                all_hazards = state.hazard_model.model._calibrator.predict(all_hazards)
            survival_df = hazard_to_survival(
                pd.Series(all_hazards),
                tenant_rows["tenant_id"],
                tenant_rows["month_of_lease"],
            )
            survival_prob = float(survival_df["survival_prob"].iloc[-1])

    driver_list = []
    if drivers:
        driver_list = [DriverDetail(**d) for d in drivers]

    narrative_drivers = []
    if driver_list:
        narrative_drivers = [driver_list[0].model_dump()]
    else:
        narrative_drivers = [{"label": "no features available", "direction": "unknown", "shap_value": 0.0}]

    narrative_result = generate_narrative(
        tenant_id, risk_pct, narrative_drivers,
        llm_fn=_ollama_generate,
    )

    return ScoreResponse(
        tenant_id=tenant_id,
        risk_pct=round(risk_pct, 1),
        risk_band=risk_band,
        survival_prob=round(survival_prob, 4) if survival_prob else None,
        top_drivers=driver_list,
        narrative=narrative_result["narrative"],
        narrative_source=narrative_result["source"],
        is_estimate=is_estimate,

    )



@app.post("/simulate", response_model=WhatIfResponse)
def simulate(req: WhatIfRequest):
    try:
        baseline_risk, baseline_drivers, _ = _score_tenant(req.tenant_id)
    except (FeatureBuildError, ModelSchemaError) as e:
        raise HTTPException(status_code=422, detail=f"Could not build features: {e}")

    scenario_risk = baseline_risk
    delta = 0.0

    if state.hazard_model is not None and "tenure_months" in state.hazard_model.feature_names:
        idx = state.hazard_model.feature_names.index("tenure_months")
        try:
            base_effect = abs(float(state.hazard_model.model.feature_importance(importance_type="gain")[idx]))
        except (AttributeError, IndexError, TypeError):
            base_effect = 1.0
    else:
        base_effect = 1.0

    lever_breakdown = []

    maint_effect = base_effect * 0.15
    if req.maintenance_speed == "priority":
        maint_contrib = maint_effect * 15
        scenario_risk -= maint_contrib
        lever_breakdown.append(LeverDetail(name="Maintenance Speed", effect_pct=-maint_contrib, direction="decreases"))
    elif req.maintenance_speed == "same_day":
        maint_contrib = maint_effect * 30
        scenario_risk -= maint_contrib
        lever_breakdown.append(LeverDetail(name="Maintenance Speed", effect_pct=-maint_contrib, direction="decreases"))
    else:
        lever_breakdown.append(LeverDetail(name="Maintenance Speed", effect_pct=0.0, direction="neutral"))

    credit_effect = base_effect * 0.008
    if req.retention_credit_usd > 0:
        credit_contrib = credit_effect * (req.retention_credit_usd / 10)
        scenario_risk -= credit_contrib
        lever_breakdown.append(LeverDetail(name="Retention Credit", effect_pct=-credit_contrib, direction="decreases"))
    else:
        lever_breakdown.append(LeverDetail(name="Retention Credit", effect_pct=0.0, direction="neutral"))

    rent_effect = base_effect * 0.04
    if req.rent_increase_pct > 0:
        rent_contrib = rent_effect * req.rent_increase_pct
        scenario_risk += rent_contrib
        lever_breakdown.append(LeverDetail(name="Rent Increase", effect_pct=rent_contrib, direction="increases"))
    else:
        lever_breakdown.append(LeverDetail(name="Rent Increase", effect_pct=0.0, direction="neutral"))

    scenario_risk = max(0.0, min(100.0, scenario_risk))
    delta = scenario_risk - baseline_risk

    if state.cate_models and state.person_period is not None:
        tenant_rows = state.person_period[
            state.person_period["tenant_id"] == req.tenant_id
        ]
        if not tenant_rows.empty:
            latest = tenant_rows.iloc[-1:]
            cate_df = state.cate_models["t_learner"].predict_cate(
                latest[["tenant_id"] + state.hazard_model.feature_names].fillna(0)
            )
            best_arm = cate_df.loc[cate_df["cate"].idxmax()] if not cate_df.empty else None
            if best_arm is not None:
                rec = f"Recommended action: {best_arm['arm']} (estimated CATE: {best_arm['cate']:.2%})"
            else:
                rec = "No specific intervention recommended based on causal analysis."
        else:
            rec = "No causal data available for this tenant."
    else:
        if delta < -5:
            rec = "Scenario reduces risk — proceed with this intervention mix."
        elif delta > 5:
            rec = "Scenario increases risk — consider a different approach."
        else:
            rec = "Scenario has minimal impact on predicted risk."

    return WhatIfResponse(
        baseline_risk_pct=round(baseline_risk, 1),
        scenario_risk_pct=round(scenario_risk, 1),
        delta_pts=round(delta, 1),
        recommendation=rec,
        narrative="",
        lever_breakdown=lever_breakdown,
    )


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    if state.cate_models is None or not state.cate_models:
        raise HTTPException(status_code=503, detail="CATE models not loaded")

    if state.tenants_df is None:
        raise HTTPException(status_code=503, detail="Tenant data not loaded")

    if state.person_period is None or state.feature_table is None:
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    latest_pp = state.person_period.loc[
        state.person_period.groupby("tenant_id")["month_of_lease"].idxmax()
    ]

    cate_df = state.cate_models["t_learner"].predict_cate(
        latest_pp[["tenant_id"] + state.hazard_model.feature_names].fillna(0)
    )

    decision_vars = []
    for arm in cate_df["arm"].unique():
        arm_cates = cate_df[cate_df["arm"] == arm]
        for _, row in arm_cates.iterrows():
            tid = row["tenant_id"]
            cate = row["cate"]
            ltv = 6 * float(
                state.tenants_df.loc[
                    state.tenants_df["tenant_id"] == tid, "household_size"
                ].iloc[0] if tid in state.tenants_df["tenant_id"].values else 1
            )
            cost = {
                "retention_credit": 150.0,
                "priority_maintenance": 75.0,
                "rent_concession": 100.0,
            }.get(arm, 100.0)
            hours = {"retention_credit": 1.0, "priority_maintenance": 3.0, "rent_concession": 0.5}.get(arm, 1.0)
            decision_vars.append({
                "tenant_id": tid,
                "action": arm,
                "cost_dollars": cost,
                "crew_hours": hours,
                "cate": max(cate, 0.01),
                "ltv": ltv,
            })

    decision_df = pd.DataFrame(decision_vars)

    try:
        result = solve_retention_allocation(
            decision_df,
            monthly_budget=req.monthly_budget,
            monthly_crew_hours=req.monthly_crew_hours,
        )
        solver_status = "optimal"
    except RuntimeError as e:
        logger.warning(f"ILP failed ({e}), falling back to greedy")
        result = solve_retention_allocation_greedy(
            decision_df,
            monthly_budget=req.monthly_budget,
            monthly_crew_hours=req.monthly_crew_hours,
        )
        solver_status = "greedy_fallback"

    total_spent = 0.0
    selections = []
    if len(result) > 0:
        merged = result.merge(
            decision_df[["tenant_id", "action", "cost_dollars"]],
            on=["tenant_id", "action"],
            how="left",
        )
        total_spent = merged["cost_dollars"].sum()
        selections = merged.to_dict(orient="records")

    return OptimizeResponse(
        selections=selections,
        total_selected=len(selections),
        total_budget_spent=float(total_spent),
        solver_status=solver_status,
    )


OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"


OLLAMA_HEADERS = {"User-Agent": "ForgeX/1.0"}


def _ollama_generate(prompt: str, timeout_s: float = 30.0) -> str:
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/generate",
        headers=OLLAMA_HEADERS,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


@app.post("/ai-search", response_model=AiSearchResponse)
def ai_search(req: AiSearchRequest):
    system_prompt = (
        "You are a helpful assistant for property managers and landlords. "
        "Answer questions based on the tenant data provided. "
        "Be concise and factual. If the data doesn't contain enough information to answer, say so."
    )
    prompt = f"{system_prompt}\n\nContext data:\n{req.context}\n\nQuestion: {req.question}\n\nAnswer:"

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{OLLAMA_BASE}/api/generate",
                headers=OLLAMA_HEADERS,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            if resp.status_code != 200:
                return AiSearchResponse(
                    answer=f"Ollama at {OLLAMA_BASE} returned {resp.status_code}. If using Cloudflare tunnel, restart Kaggle and get a fresh URL.",
                    source="error",
                )
            data = resp.json()
            return AiSearchResponse(
                answer=data.get("response", "No response from model"),
                source="ollama",
            )
    except httpx.ConnectError:
        return AiSearchResponse(
            answer="Cannot reach Ollama at the configured URL. If using Cloudflare tunnel, restart Kaggle and share the new URL.",
            source="error",
        )
    except Exception as e:
        return AiSearchResponse(
            answer=f"AI search error: {str(e)}",
            source="error",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "forgex.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )




