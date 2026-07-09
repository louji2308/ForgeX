from __future__ import annotations

import numpy as np
import pandas as pd

from forgex.logging_setup import get_logger

logger = get_logger(__name__)


def inject_historical_bias(
    tenants: pd.DataFrame,
    leases: pd.DataFrame,
    payments: pd.DataFrame,
    rng: np.random.Generator,
    bias_strength: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Injects a deliberate historical bias scenario: some legacy risk flags
    were historically harsher for voucher-holder-correlated tenants despite
    no true difference in payment behavior.

    Returns:
        bias_ground_truth: DataFrame with the injected bias labels.
        biased_leases: Copy of leases with added 'legacy_risk_flag' column.
    """
    biased_leases = leases.copy()

    # Compute a biased risk score that penalizes voucher holders and
    # tenants in certain zip codes, regardless of actual payment behavior
    tenant_voucher = tenants.set_index("tenant_id")["voucher_holder"].to_dict()
    tenant_zip = tenants.set_index("tenant_id")["zip_code"].to_dict()

    biased_flags = []
    for _, lease in biased_leases.iterrows():
        tid = lease["tenant_id"]
        is_voucher = tenant_voucher.get(tid, False)
        zip_code = tenant_zip.get(tid, "ZIP_0000")

        # Base risk on actual late payments
        lease_payments = payments[payments["lease_id"] == lease["lease_id"]]
        late_rate = lease_payments["is_late"].mean() if len(lease_payments) > 0 else 0
        late_severity = lease_payments["days_late"].mean() if len(lease_payments) > 0 else 0

        # Bias: add extra risk for voucher holders and certain zip codes
        bias_penalty = 0.0
        if is_voucher:
            bias_penalty += bias_strength * 0.4
        if zip_code in {"ZIP_0003", "ZIP_0007", "ZIP_0012", "ZIP_0019"}:
            bias_penalty += bias_strength * 0.2

        combined_risk = np.clip(late_rate * 0.6 + late_severity * 0.02 + bias_penalty + rng.normal(0, 0.05), 0, 1)
        flagged = combined_risk > 0.35
        biased_flags.append(flagged)

    biased_leases["legacy_risk_flag"] = biased_flags
    biased_leases["_true_risk_no_bias"] = 0.0

    bias_ground_truth = pd.DataFrame({
        "tenant_id": biased_leases["tenant_id"].unique(),
        "was_flagged_by_legacy_bias": biased_leases.groupby("tenant_id")["legacy_risk_flag"].first().values,
    })

    logger.info(
        f"Bias injection complete: {bias_ground_truth['was_flagged_by_legacy_bias'].mean():.1%} "
        f"tenants flagged by legacy biased system"
    )

    return bias_ground_truth, biased_leases
