from __future__ import annotations

import numpy as np
import pandas as pd

from forgex.config import load_settings
from forgex.logging_setup import get_logger
from forgex.simulation.bias_injection import inject_historical_bias
from forgex.simulation.entities import generate_tenants, generate_properties, generate_units
from forgex.simulation.events import generate_lease, generate_monthly_events
from forgex.simulation.hidden_state import evolve_hidden_states
from forgex.simulation.market import generate_market_comps
from forgex.simulation.validation import validate_synthetic_world

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable, **kwargs):
        return iterable

logger = get_logger(__name__)


class SyntheticWorldEngine:
    """Orchestrates the full synthetic data generation pipeline.
    Every model downstream is only as honest as this data is — this engine
    builds a world with real causal structure, not a numpy.random dump."""

    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        self.cfg = self.settings.data
        self._rng = np.random.default_rng(self.cfg.seed)
        self._start = pd.Timestamp(self.cfg.start_date)
        self._end = pd.Timestamp(self.cfg.end_date)

    def generate(self) -> dict[str, pd.DataFrame]:
        logger.info(
            f"Generating synthetic world: {self.cfg.n_tenants} tenants, "
            f"{self.cfg.n_properties} properties, "
            f"{self._start.date()} to {self._end.date()}"
        )

        tenants = generate_tenants(self.cfg.n_tenants, self._rng)
        logger.info(f"Generated {len(tenants)} tenants with hidden segments")

        properties = generate_properties(self.cfg.n_properties, self._rng)
        units = generate_units(properties, self._rng)
        logger.info(f"Generated {len(properties)} properties with {len(units)} units")

        market_comps = generate_market_comps(
            properties, self._start - pd.DateOffset(years=1), self._end, self._rng
        )

        all_leases: list[dict] = []
        all_payments: list[dict] = []
        all_maintenance: list[dict] = []
        all_interventions: list[dict] = []
        hidden_segments_records: list[dict] = []

        n_tenant_months = 0
        total_renewals = 0
        total_leases = 0

        lease_counter = 0
        for _, tenant in _tqdm(tenants.iterrows(), total=len(tenants), desc="Generating tenants", unit="tenant"):
            n_leases = self._rng.integers(1, 4)
            for lease_num in range(n_leases):
                unit = units.iloc[self._rng.integers(0, len(units))]
                duration = int(self._rng.integers(6, 25))
                lease_start = self._start + pd.DateOffset(
                    months=int(self._rng.integers(0, 36)) + lease_num * duration
                )
                if lease_start >= self._end:
                    continue

                lease_counter += 1
                lease = generate_lease(tenant, unit, lease_start, duration, lease_counter, self._rng)
                tenure_months = min(duration, int((self._end - lease_start).days / 30))
                if tenure_months < 1:
                    continue

                nbh = unit["neighborhood_cluster"]
                market_rent_row = market_comps[
                    (market_comps["neighborhood_cluster"] == nbh)
                    & (market_comps["month"] == lease_start)
                ]
                market_rent = float(market_rent_row["market_rent"].iloc[0]) if len(market_rent_row) > 0 else float(unit["base_rent"])

                hidden_states = evolve_hidden_states(tenant, tenure_months, lease_start, self._rng)

                payments, maintenance, interventions, churned, censored = generate_monthly_events(
                    tenant, unit, lease, hidden_states, market_rent, 0, tenure_months, self._rng,
                )

                lease["did_renew"] = not churned
                lease["is_censored"] = censored
                all_leases.append(lease)
                all_payments.extend(payments)
                all_maintenance.extend(maintenance)
                all_interventions.extend(interventions)
                n_tenant_months += tenure_months
                total_renewals += int(not churned)
                total_leases += 1

                hidden_segments_records.append({
                    "tenant_id": tenant.tenant_id,
                    "true_segment": str(tenant["hidden_segment"]),
                })

        leases_df = pd.DataFrame(all_leases)
        payments_df = pd.DataFrame(all_payments)
        maintenance_df = pd.DataFrame(all_maintenance)
        intervention_df = pd.DataFrame(all_interventions)
        hidden_segments_df = pd.DataFrame(hidden_segments_records).drop_duplicates("tenant_id")

        bias_ground_truth, biased_leases = inject_historical_bias(
            tenants, leases_df, payments_df, self._rng
        )

        tables = {
            "tenants": tenants.drop(columns=["_satisfaction_base", "_financial_stress_base", "hidden_segment"], errors="ignore"),
            "properties": properties,
            "units": units,
            "leases": biased_leases,
            "payments": payments_df,
            "maintenance_requests": maintenance_df,
            "market_comps": market_comps,
            "intervention_log": intervention_df,
        }

        validate_synthetic_world(tables)

        renewal_rate = total_renewals / max(total_leases, 1)
        logger.info(
            f"Generation complete: {total_leases} leases, "
            f"{n_tenant_months} tenant-months, "
            f"renewal rate {renewal_rate:.1%}"
        )

        if not (0.55 <= renewal_rate <= 0.85):
            logger.warning(
                f"Renewal rate {renewal_rate:.1%} outside expected 55-85% band — "
                f"DGP weights may need recalibrating"
            )

        return tables, hidden_segments_df, bias_ground_truth

    def regenerate(self) -> dict[str, pd.DataFrame]:
        self._rng = np.random.default_rng(self.cfg.seed)
        return self.generate()


def run_simulation(settings=None) -> dict[str, pd.DataFrame]:
    engine = SyntheticWorldEngine(settings)
    return engine.generate()
