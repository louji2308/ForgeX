from __future__ import annotations

import numpy as np
import pandas as pd

from forgex.simulation.utils import assign_hidden_segments


def generate_tenants(n_tenants: int, rng: np.random.Generator) -> pd.DataFrame:
    household_size = rng.choice([1, 2, 3, 4, 5, 6], size=n_tenants, p=[0.25, 0.35, 0.20, 0.12, 0.05, 0.03])
    voucher_holder = rng.random(n_tenants) < 0.15
    zip_codes = rng.choice([f"ZIP_{i:04d}" for i in range(1, 21)], size=n_tenants)

    tenants = pd.DataFrame({
        "tenant_id": [f"T{i:06d}" for i in range(n_tenants)],
        "household_size": household_size,
        "voucher_holder": voucher_holder,
        "zip_code": zip_codes,
        "move_in_date": pd.Timestamp("2017-01-01") + pd.to_timedelta(
            rng.integers(0, 730, size=n_tenants), unit="D"
        ),
    })

    hidden_segments = assign_hidden_segments(
        household_size.astype(float), rng
    )

    satisfaction_base = np.clip(
        0.6 + 0.05 * (4 - household_size) + rng.normal(0, 0.15, size=n_tenants),
        0.1, 0.95,
    )
    financial_stress_base = np.clip(
        0.15 + 0.08 * (household_size > 3).astype(float) + rng.normal(0, 0.1, size=n_tenants),
        0.01, 0.9,
    )

    tenants["hidden_segment"] = hidden_segments
    tenants["_satisfaction_base"] = satisfaction_base
    tenants["_financial_stress_base"] = financial_stress_base
    return tenants


def generate_properties(n_properties: int, rng: np.random.Generator) -> pd.DataFrame:
    neighborhoods = [f"NBH_{i:02d}" for i in range(10)]
    property_types = ["apartment", "house", "condo", "townhouse"]

    properties = pd.DataFrame({
        "property_id": [f"P{i:04d}" for i in range(n_properties)],
        "neighborhood_cluster": rng.choice(neighborhoods, size=n_properties),
        "property_type": rng.choice(property_types, size=n_properties, p=[0.5, 0.2, 0.2, 0.1]),
        "year_built": rng.integers(1960, 2024, size=n_properties),
        "units_per_property": rng.poisson(lam=8, size=n_properties).clip(1, 50),
    })
    return properties


def generate_units(properties: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    units = []
    for _, prop in properties.iterrows():
        for _ in range(int(prop["units_per_property"])):
            sqft = rng.integers(400, 2000)
            bedrooms = rng.choice([0, 1, 2, 3, 4], p=[0.02, 0.30, 0.40, 0.22, 0.06])
            base_rent = np.clip(
                800 + sqft * 1.5 + bedrooms * 200 + rng.normal(0, 100),
                400, 5000,
            )
            units.append({
                "unit_id": f"U{len(units):07d}",
                "property_id": prop["property_id"],
                "neighborhood_cluster": prop["neighborhood_cluster"],
                "sqft": sqft,
                "bedrooms": bedrooms,
                "base_rent": round(base_rent, 2),
            })
    return pd.DataFrame(units)
