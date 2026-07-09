from __future__ import annotations

import numpy as np
import pandas as pd

from forgex.simulation.utils import (
    compute_renewal_probability,
)
from forgex.logging_setup import get_logger

logger = get_logger(__name__)

MAINTENANCE_TEXTS = {
    "critical": [
        "No heat in unit for 3 days, temperature dropping below 50",
        "Mold growing in bathroom walls, children having asthma attacks",
        "No hot water for a week, cannot shower",
        "Gas smell coming from the kitchen stove",
        "Flooding from burst pipe, water damaging our belongings",
        "Not cooling at all, AC completely dead in 95 degree weather",
    ],
    "moderate": [
        "Leak under kitchen sink, cabinet is damaged",
        "Dishwasher not working, needs repair",
        "Broken garbage disposal, smells bad",
        "Clogged toilet in main bathroom",
        "Not working properly, refrigerator is warm",
    ],
    "minor": [
        "Cosmetic crack in the ceiling, no leak",
        "Squeaky door hinge in bedroom",
        "Loose handle on kitchen cabinet",
        "Peeling paint in hallway, purely cosmetic",
        "Screen door has small tear",
    ],
}


def _generate_maintenance_text(severity: str, rng: np.random.Generator) -> str:
    texts = MAINTENANCE_TEXTS.get(severity, MAINTENANCE_TEXTS["minor"])
    return str(rng.choice(texts))


def generate_lease(
    tenant: pd.Series,
    unit: pd.Series,
    lease_start: pd.Timestamp,
    lease_duration_months: int,
    lease_counter: int,
    rng: np.random.Generator,
) -> dict:
    lease_end = lease_start + pd.DateOffset(months=lease_duration_months)
    rent = float(unit["base_rent"])
    return {
        "lease_id": f"L{lease_counter:09d}",
        "tenant_id": tenant.tenant_id,
        "unit_id": unit.unit_id,
        "property_id": unit.property_id,
        "lease_start": lease_start,
        "lease_end": lease_end,
        "initial_rent": rent,
        "current_rent": rent,
        "lease_duration_months": lease_duration_months,
        "did_renew": True,
    }


def generate_monthly_events(
    tenant: pd.Series,
    unit: pd.Series,
    lease: dict,
    hidden_states: pd.DataFrame,
    market_rent: float,
    start_month: int,
    n_months: int,
    rng: np.random.Generator,
    weights: dict[str, float] | None = None,
) -> tuple[list[dict], list[dict], list[dict], bool, bool]:
    """Generates all events for a single lease month by month.
    Returns (payments, maintenance_requests, intervention_log entries,
             churned, was_censored)."""
    payments: list[dict] = []
    maintenance_requests: list[dict] = []
    intervention_log: list[dict] = []

    monthly_rent = float(lease["current_rent"])
    tenure_at_start = max(0, int(
        (lease["lease_start"] - tenant["move_in_date"]).days / 30
    ))

    for i in range(n_months):
        state = hidden_states.iloc[i]
        month_date = lease["lease_start"] + pd.DateOffset(months=i)
        tenure_months = tenure_at_start + i

        # --- Payment behavior driven by hidden state ---
        stress = float(state["financial_stress"])
        days_late = max(0, int(rng.poisson(max(0, stress * 15 - 2))))
        is_late = days_late > 0
        amount_paid = monthly_rent if not is_late else monthly_rent * (1 - 0.05 * rng.random())

        payments.append({
            "tenant_id": tenant.tenant_id,
            "lease_id": lease["lease_id"],
            "month": month_date,
            "amount_due": monthly_rent,
            "amount_paid": round(amount_paid, 2),
            "days_late": days_late,
            "is_late": is_late,
        })

        # --- Maintenance requests driven by hidden state ---
        sat = float(state["satisfaction"])
        maintenance_prob = 0.03 + 0.04 * (1 - sat) + 0.02 * stress
        if rng.random() < maintenance_prob:
            if stress > 0.6 or sat < 0.2:
                severity = rng.choice(["critical", "moderate", "minor"], p=[0.5, 0.3, 0.2])
            elif sat < 0.4:
                severity = rng.choice(["moderate", "minor"], p=[0.5, 0.5])
            else:
                severity = rng.choice(["moderate", "minor"], p=[0.2, 0.8])

            complaint_text = _generate_maintenance_text(severity, rng)

            maintenance_requests.append({
                "tenant_id": tenant.tenant_id,
                "lease_id": lease["lease_id"],
                "unit_id": unit["unit_id"],
                "request_date": month_date,
                "text": complaint_text,
                "severity_ground_truth": severity,
                "days_to_resolve": int(rng.exponential(scale=5) + 1),
            })

    # --- Renewal decision ---
    final_state = hidden_states.iloc[-1]
    final_sat = float(final_state["satisfaction"])
    final_stress = float(final_state["financial_stress"])
    tenure_months_arr = np.array([tenure_at_start + n_months - 1])

    renewal_p = compute_renewal_probability(
        cumulative_satisfaction=np.array([final_sat]),
        financial_stress_at_decision=np.array([final_stress]),
        rent_increase_pct=np.array([0.0]),
        market_gap_pct=np.array([max(0, monthly_rent - market_rent) / monthly_rent]),
        tenure_months=tenure_months_arr,
        rng=rng,
        weights=weights,
    )[0]

    did_renew = rng.random() < renewal_p
    churned = not did_renew
    was_censored = False

    # --- Interventions (historic offers triggered by risk signals) ---
    # Generate regardless of churn outcome: at-risk tenants get offered help.
    # Risk score = low satisfaction + high stress (range ~0–1.5)
    risk_score = (1 - final_sat) + min(final_stress, 1.0)
    if rng.random() < 0.15 + 0.35 * risk_score:
        iv_type = rng.choice(["retention_credit", "priority_maintenance", "rent_concession"])
        iv_cost = {"retention_credit": rng.uniform(50, 300),
                    "priority_maintenance": rng.uniform(0, 150),
                    "rent_concession": rng.uniform(30, 200)}[iv_type]
        iv_hours = {"retention_credit": 1.0,
                     "priority_maintenance": rng.uniform(2, 6),
                     "rent_concession": 0.5}[iv_type]
        intervention_log.append({
            "tenant_id": tenant.tenant_id,
            "lease_id": lease["lease_id"],
            "intervention_type": iv_type,
            "cost_dollars": round(iv_cost, 2),
            "crew_hours": round(iv_hours, 2),
            "offered_date": lease["lease_end"] - pd.DateOffset(months=2),
            "accepted": bool(rng.random() < 0.6),
            "renewed": did_renew,
        })

    return payments, maintenance_requests, intervention_log, churned, was_censored
