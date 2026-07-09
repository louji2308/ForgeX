from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forgex.config import load_settings
from forgex.logging_setup import get_logger

logger = get_logger(__name__)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """PSI < 0.1: no significant shift. 0.1–0.25: moderate, watch it.
    > 0.25: significant — investigate before trusting new predictions."""
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("reference and current must both be non-empty")

    bin_edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        raise ValueError(
            "Reference distribution too degenerate to bin (too many tied values)"
        )

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-4, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-4, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_drift(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    psi_threshold: float = 0.25,
    max_numeric_features: int | None = None,
) -> dict:
    common_cols = set(reference_features.columns) & set(current_features.columns)
    if not common_cols:
        raise ValueError(
            "No overlapping columns between reference and current feature sets"
        )

    results = {}
    skipped = 0
    for i, col in enumerate(common_cols):
        if max_numeric_features is not None and i >= max_numeric_features:
            break
        if not pd.api.types.is_numeric_dtype(reference_features[col]):
            continue
        try:
            ref_vals = reference_features[col].dropna().values
            cur_vals = current_features[col].dropna().values
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                skipped += 1
                continue
            results[col] = population_stability_index(ref_vals, cur_vals)
        except ValueError as e:
            logger.warning(f"Skipping PSI for '{col}': {e}")
            skipped += 1

    breached = {k: v for k, v in results.items() if v > psi_threshold}

    if skipped > 0:
        logger.info(f"Skipped {skipped} features due to insufficient data")

    return {
        "psi_by_feature": results,
        "breached_features": breached,
        "retrain_recommended": len(breached) > 0,
        "features_checked": len(results),
        "features_breached": len(breached),
    }


class DriftMonitor:
    """Continuously monitors feature distributions and triggers alerts
    when Population Stability Index exceeds the configured threshold."""

    def __init__(self, reference_features: pd.DataFrame, psi_threshold: float = 0.25):
        self.reference_features = reference_features
        self.psi_threshold = psi_threshold
        self._drift_history: list[dict] = []

    def check(self, current_features: pd.DataFrame) -> dict:
        report = check_drift(
            self.reference_features,
            current_features,
            psi_threshold=self.psi_threshold,
        )
        self._drift_history.append(report)

        n_breached = len(report["breached_features"])
        if n_breached > 0:
            top_breached = sorted(
                report["breached_features"].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
            logger.warning(
                f"Drift detected: {n_breached} features breached threshold "
                f"(PSI > {self.psi_threshold}). Top: {top_breached}"
            )

        return report

    def generate_alert(self, report: dict | None = None) -> str:
        r = report or (self._drift_history[-1] if self._drift_history else None)
        if r is None:
            return "No drift data available."

        if r["retrain_recommended"]:
            top_features = list(r["breached_features"].items())[:3]
            return (
                f"RETRAIN RECOMMENDED: {r['features_breached']}/{r['features_checked']} "
                f"features exhibit significant drift (PSI > {self.psi_threshold}). "
                f"Most drifted: {top_features}. Model predictions may no longer be reliable."
            )
        return (
            f"All clear: {r['features_checked']} features checked, "
            f"none exceeded PSI threshold of {self.psi_threshold}."
        )

    @property
    def drift_history(self) -> list[dict]:
        return list(self._drift_history)
