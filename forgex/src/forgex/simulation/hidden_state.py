from __future__ import annotations

import numpy as np
import pandas as pd

from forgex.logging_setup import get_logger

logger = get_logger(__name__)


def evolve_hidden_states(
    tenant: pd.Series,
    n_months: int,
    base_date: pd.Timestamp,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Evolves satisfaction and financial_stress month by month for one tenant.
    Returns a DataFrame with one row per month containing the hidden states
    that drive every observable signal — never exposed as features."""
    records = []
    sat = float(tenant["_satisfaction_base"])
    stress = float(tenant["_financial_stress_base"])

    for month_offset in range(n_months):
        current_date = base_date + pd.DateOffset(months=month_offset)

        # Satisfaction drifts with small noise
        sat += rng.normal(0, 0.02)

        # Maintenance issues erode satisfaction
        has_issue = rng.random() < 0.08
        if has_issue:
            sat -= rng.uniform(0.02, 0.12)

        # Life-event shock: low-probability events that are structurally
        # unobservable from landlord data — modeling irreducible churn
        # honestly is a credibility signal, not a weakness.
        life_event = rng.random() < 0.015
        if life_event:
            sat -= rng.uniform(0.1, 0.3)
            stress += rng.uniform(0.05, 0.2)

        # Rent increases drive stress
        rent_increase_event = rng.random() < 0.04
        if rent_increase_event:
            stress += rng.uniform(0.03, 0.10)

        # Stress mean-reverts slowly
        stress += 0.02 * (tenant["_financial_stress_base"] - stress) + rng.normal(0, 0.01)

        sat = float(np.clip(sat, 0.01, 0.99))
        stress = float(np.clip(stress, 0.01, 0.99))

        records.append({
            "month": current_date,
            "month_offset": month_offset,
            "satisfaction": sat,
            "financial_stress": stress,
            "life_event_shock": life_event,
            "had_maintenance_issue": has_issue,
        })

    return pd.DataFrame(records)
