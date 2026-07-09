from __future__ import annotations

import numpy as np
import pandas as pd


def generate_market_comps(
    properties: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generates monthly market rent per neighborhood cluster with
    realistic seasonality and drift."""
    months = pd.date_range(start_date, end_date, freq="MS")
    neighborhoods = properties["neighborhood_cluster"].unique()

    records = []
    for nbh in neighborhoods:
        base = 1200 + rng.uniform(0, 600)
        growth_trend = rng.uniform(0.002, 0.005)

        for i, m in enumerate(months):
            seasonality = 1.0 + 0.05 * np.sin(2 * np.pi * i / 12)
            market_rent = round(base * (1 + growth_trend) ** i * seasonality + rng.normal(0, 30), 2)
            records.append({
                "neighborhood_cluster": nbh,
                "month": m,
                "market_rent": max(400, market_rent),
            })

    return pd.DataFrame(records)


def generate_intervention_log(
    lease_id: str,
    tenant_id: str,
    intervention_type: str,
    cost_dollars: float,
    crew_hours: float,
    offered_date: pd.Timestamp,
    accepted: bool,
    renewed: bool,
) -> dict:
    return {
        "intervention_id": f"I{hash(f'{lease_id}{tenant_id}{offered_date}') % 10_000_000:07d}",
        "lease_id": lease_id,
        "tenant_id": tenant_id,
        "intervention_type": intervention_type,
        "cost_dollars": cost_dollars,
        "crew_hours": crew_hours,
        "offered_date": offered_date,
        "accepted": accepted,
        "renewed": renewed,
    }
