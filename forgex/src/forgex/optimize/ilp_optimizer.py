from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pulp

from forgex.logging_setup import get_logger

logger = get_logger(__name__)


def solve_retention_allocation(
    tenants: pd.DataFrame,
    monthly_budget: float,
    monthly_crew_hours: float,
    max_actions_per_tenant: int = 1,
    solver_time_limit_s: int = 30,
    solver_msg: bool = False,
) -> pd.DataFrame:
    """Integer Linear Program for budget-constrained retention allocation.

    Required columns: tenant_id, action, cost_dollars, crew_hours, cate, ltv.
    Maximizes Σ CATE_i × LTV_i × x_i subject to budget and crew-hour caps.
    """
    if tenants.empty:
        raise ValueError("No tenants provided to optimizer")
    if monthly_budget <= 0:
        raise ValueError(f"monthly_budget must be positive, got {monthly_budget}")

    required = {"tenant_id", "action", "cost_dollars", "crew_hours", "cate", "ltv"}
    if missing := required - set(tenants.columns):
        raise ValueError(f"tenants missing required columns: {missing}")

    prob = pulp.LpProblem("retention_allocation", pulp.LpMaximize)

    x = {}
    for r in tenants.itertuples():
        x[(r.tenant_id, r.action)] = pulp.LpVariable(
            f"x_{r.tenant_id}_{r.action}", cat="Binary"
        )

    # Objective: maximize total CATE × LTV
    prob += pulp.lpSum(
        x[(r.tenant_id, r.action)] * float(r.cate) * float(r.ltv)
        for r in tenants.itertuples()
    )

    # Budget constraint
    prob += (
        pulp.lpSum(
            x[(r.tenant_id, r.action)] * float(r.cost_dollars)
            for r in tenants.itertuples()
        ) <= monthly_budget,
        "budget_cap",
    )

    # Crew-hour constraint
    prob += (
        pulp.lpSum(
            x[(r.tenant_id, r.action)] * float(r.crew_hours)
            for r in tenants.itertuples()
        ) <= monthly_crew_hours,
        "crew_cap",
    )

    # At most one action per tenant
    for tenant_id, group in tenants.groupby("tenant_id"):
        prob += (
            pulp.lpSum(x[(tenant_id, a)] for a in group["action"]) <= max_actions_per_tenant,
            f"one_action_{tenant_id}",
        )

    prob.solve(
        pulp.PULP_CBC_CMD(msg=solver_msg, timeLimit=solver_time_limit_s)
    )

    status = pulp.LpStatus[prob.status]

    if status == "Infeasible":
        raise RuntimeError(
            "Optimizer is infeasible with current constraints. Likely cause: "
            "budget/crew-hour cap too small for even one action. Try raising monthly_budget."
        )

    if status != "Optimal":
        logger.warning(
            f"Solver status='{status}' — result may be a best-effort bound, not the true optimum."
        )

    selected = [
        {"tenant_id": t, "action": a, "variable_value": var.value()}
        for (t, a), var in x.items() if var.value() == 1
    ]
    result = pd.DataFrame(selected)

    total_spend = result.merge(
        tenants[["tenant_id", "action", "cost_dollars", "crew_hours", "cate", "ltv"]],
        on=["tenant_id", "action"],
        how="left"
    )["cost_dollars"].sum() if len(result) > 0 else 0.0

    logger.info(
        f"Optimizer status={status}, selected {len(result)}/{len(tenants)} actions, "
        f"total spend=${total_spend:.2f}"
    )

    return result


def solve_retention_allocation_greedy(
    tenants: pd.DataFrame,
    monthly_budget: float,
    monthly_crew_hours: float,
) -> pd.DataFrame:
    """Fallback for solver timeouts on stage. Provably suboptimal, never
    hangs, always returns something to show."""
    ranked = (
        tenants.assign(
            vpd=lambda d: (d["cate"] * d["ltv"]) / d["cost_dollars"].clip(lower=0.01)
        )
        .sort_values("vpd", ascending=False)
    )

    chosen = []
    spend = 0.0
    hours = 0.0
    used_tenants = set()

    for r in ranked.itertuples():
        if r.tenant_id in used_tenants:
            continue
        if (
            spend + r.cost_dollars <= monthly_budget
            and hours + r.crew_hours <= monthly_crew_hours
        ):
            chosen.append(r.tenant_id)
            spend += r.cost_dollars
            hours += r.crew_hours
            used_tenants.add(r.tenant_id)

    logger.info(
        f"Greedy optimizer: selected {len(chosen)} tenants, "
        f"spend=${spend:.2f}, hours={hours:.1f}"
    )

    return tenants[tenants["tenant_id"].isin(chosen)]
