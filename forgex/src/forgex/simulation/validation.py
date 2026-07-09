from __future__ import annotations

import pandas as pd

from forgex.errors import DataValidationError


def validate_synthetic_world(tables: dict[str, pd.DataFrame]) -> None:
    errors: list[str] = []
    leases, tenants, units = tables["leases"], tables["tenants"], tables["units"]
    payments = tables["payments"]

    required_tables = ["tenants", "units", "leases", "payments", "maintenance_requests",
                        "market_comps", "intervention_log"]
    for name in required_tables:
        if name not in tables:
            errors.append(f"Missing required table: {name}")
            continue
        df = tables[name]
        if df.empty:
            errors.append(f"Table '{name}' is empty")

    if errors:
        raise DataValidationError(
            "Synthetic world failed validation:\n  - " + "\n  - ".join(errors)
        )

    orphan_tenant = leases.loc[~leases["tenant_id"].isin(tenants["tenant_id"])]
    if len(orphan_tenant):
        errors.append(f"{len(orphan_tenant)} leases reference unknown tenant_id")

    orphan_unit = leases.loc[~leases["unit_id"].isin(units["unit_id"])]
    if len(orphan_unit):
        errors.append(f"{len(orphan_unit)} leases reference unknown unit_id")

    bad_dates = leases.loc[leases["lease_start"] >= leases["lease_end"]]
    if len(bad_dates):
        errors.append(f"{len(bad_dates)} leases have lease_start >= lease_end")

    if (units["base_rent"] <= 0).any():
        errors.append("base_rent contains non-positive values")

    if payments["amount_paid"].lt(0).any():
        errors.append("payments contain negative amount_paid")

    for name, df, key in [
        ("tenants", tenants, "tenant_id"),
        ("units", units, "unit_id"),
        ("leases", leases, "lease_id"),
    ]:
        dupes = df[key].duplicated().sum()
        if dupes:
            errors.append(f"{name}: {dupes} duplicate {key} values")

    if errors:
        raise DataValidationError(
            "Synthetic world failed validation:\n  - " + "\n  - ".join(errors)
        )
