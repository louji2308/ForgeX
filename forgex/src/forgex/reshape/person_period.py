from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forgex.errors import DataValidationError
from forgex.logging_setup import get_logger

logger = get_logger(__name__)


def explode_to_person_period(
    leases: pd.DataFrame, as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per (tenant, lease-month). Required columns: tenant_id,
    lease_id, lease_start, lease_end, did_renew."""
    required = {"tenant_id", "lease_id", "lease_start", "lease_end", "did_renew"}
    if missing := required - set(leases.columns):
        raise DataValidationError(f"leases missing required columns: {missing}")

    if as_of_date is None:
        as_of_date = leases["lease_end"].max()

    rows: list[dict] = []
    for lease in leases.itertuples(index=False):
        if lease.lease_start >= lease.lease_end:
            raise DataValidationError(f"lease {lease.lease_id}: lease_start >= lease_end")

        observed_end = min(lease.lease_end, as_of_date)
        is_censored = bool(as_of_date < lease.lease_end)

        months = pd.date_range(lease.lease_start, observed_end, freq=pd.DateOffset(months=1))
        if len(months) == 0:
            continue

        for i, month_start in enumerate(months, start=1):
            is_final_row = i == len(months)
            churn_event = int(is_final_row and not is_censored and not bool(lease.did_renew))
            rows.append({
                "tenant_id": lease.tenant_id,
                "lease_id": lease.lease_id,
                "month_of_lease": i,
                "calendar_month": month_start,
                "still_active": 1,
                "churn_event_this_month": churn_event,
                "is_censored": int(is_final_row and is_censored),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise DataValidationError("Reshape produced zero rows — check as_of_date vs lease ranges")
    return out


def validate_person_period(person_period: pd.DataFrame, leases: pd.DataFrame) -> None:
    """Row-count conservation is the cheapest, highest-signal check in a
    survival pipeline: if it fails here, nothing downstream can be trusted."""
    missing_leases = set(leases["lease_id"]) - set(person_period["lease_id"])
    if missing_leases:
        raise DataValidationError(
            f"{len(missing_leases)} leases produced zero person-period rows"
        )

    multi_event = person_period.groupby("lease_id")["churn_event_this_month"].sum()
    if len(bad := multi_event[multi_event > 1]):
        raise DataValidationError(
            f"{len(bad)} leases have more than one churn-event row"
        )

    for lease_id, group in person_period.groupby("lease_id"):
        g = group.sort_values("month_of_lease")
        if g["churn_event_this_month"].iloc[:-1].sum() > 0:
            raise DataValidationError(
                f"lease {lease_id}: churn event flagged before final month"
            )

    censored_churn = person_period[
        (person_period["is_censored"] == 1) & (person_period["churn_event_this_month"] == 1)
    ]
    if len(censored_churn):
        raise DataValidationError(
            f"{len(censored_churn)} censored rows have churn_event_this_month=1 — "
            f"a censored tenant is not the same as a churned tenant"
        )


def tenant_level_split(
    tenant_ids: pd.Series,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[set, set, set]:
    """Leak-proof split by tenant_id — all months of one tenant must land
    in the same fold or you leak tenant identity across the split."""
    unique_tenants = tenant_ids.unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_tenants)
    n = len(unique_tenants)
    test_ids = set(unique_tenants[: int(n * test_frac)])
    val_ids = set(unique_tenants[int(n * test_frac): int(n * (test_frac + val_frac))])
    train_ids = set(unique_tenants) - test_ids - val_ids

    assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids), \
        "Split produced overlapping tenant_ids — this WILL leak"
    return train_ids, val_ids, test_ids


def reshape_pipeline(
    tables: dict[str, pd.DataFrame],
    feature_table: pd.DataFrame | None = None,
    as_of_date: pd.Timestamp | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict:
    """End-to-end person-period reshape with feature join and split."""
    leases = tables["leases"]

    logger.info(f"Exploding {len(leases)} leases to person-period format...")
    person_period = explode_to_person_period(leases, as_of_date)
    logger.info(f"Generated {len(person_period)} person-month rows")

    validate_person_period(person_period, leases)
    logger.info("Person-period validation passed")

    if feature_table is not None:
        logger.info("Joining point-in-time features...")
        feature_table["calendar_month"] = pd.to_datetime(feature_table["as_of_month"])
        # Deduplicate: a tenant can have multiple active leases in a month,
        # producing duplicate (tenant_id, calendar_month) rows in the feature table.
        numeric_cols = feature_table.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in feature_table.columns
                            if c not in numeric_cols and c not in ("tenant_id", "calendar_month")]
        aggs = {c: "mean" for c in numeric_cols} | {c: "first" for c in non_numeric_cols}
        if aggs:
            feature_table = feature_table.groupby(["tenant_id", "calendar_month"], as_index=False).agg(aggs)
        pre_join = len(person_period)
        person_period = person_period.merge(
            feature_table,
            on=["tenant_id", "calendar_month"],
            how="left",
        )
        if len(person_period) != pre_join:
            raise DataValidationError(
                f"Feature join changed row count from {pre_join} to {len(person_period)} — "
                f"check for duplicate merge keys"
            )
        logger.info("Feature join complete")

    train_ids, val_ids, test_ids = tenant_level_split(
        leases["tenant_id"], seed=seed
    )
    logger.info(
        f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test tenants"
    )

    person_period["fold"] = "unassigned"
    person_period.loc[person_period["tenant_id"].isin(train_ids), "fold"] = "train"
    person_period.loc[person_period["tenant_id"].isin(val_ids), "fold"] = "val"
    person_period.loc[person_period["tenant_id"].isin(test_ids), "fold"] = "test"

    unassigned = person_period[person_period["fold"] == "unassigned"]
    if len(unassigned):
        raise DataValidationError(
            f"{len(unassigned)} rows with unassigned fold — split mismatch"
        )

    result = {
        "person_period": person_period,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        person_period.to_parquet(output_path / "person_period.parquet", index=False)
        split_info = {
            "train_ids": list(train_ids),
            "val_ids": list(val_ids),
            "test_ids": list(test_ids),
        }
        with open(output_path / "split_ids.json", "w") as f:
            json.dump(split_info, f)
        logger.info(f"Saved person-period data to {output_path}")

    return result
