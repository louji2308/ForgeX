from __future__ import annotations

import numpy as np
import pandas as pd

from forgex.errors import FeatureBuildError, DataValidationError
from forgex.features.nlp import tag_maintenance_text
from forgex.logging_setup import get_logger

logger = get_logger(__name__)


def compute_point_in_time_features(
    events: pd.DataFrame,
    as_of_month: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    """Aggregates events up to but NOT including as_of_month, using only
    rows strictly before the cutoff. This boundary is the entire difference
    between a legitimate forecasting model and one that silently cheats by
    seeing its own future."""
    if pd.isna(events["event_date"].max()) if "event_date" in events.columns else True:
        raise FeatureBuildError("events.event_date contains no valid timestamps")

    window_start = as_of_month - pd.Timedelta(days=window_days)
    in_window = events[
        (events["event_date"] >= window_start) & (events["event_date"] < as_of_month)
    ]
    return in_window


class FeaturePipeline:
    """Transforms raw event streams into point-in-time-correct features.
    Treats the cutoff boundary as sacred — target leakage is born or
    prevented here."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.tables = tables

    def _build_payment_features(self, as_of: pd.Timestamp) -> pd.DataFrame:
        payments = self.tables["payments"].copy()
        payments["event_date"] = payments["month"]

        features_list = []
        for window in [30, 60, 90]:
            window_events = compute_point_in_time_features(payments, as_of, window)
            if window_events.empty:
                continue

            agg = window_events.groupby("tenant_id").agg(
                **{
                    f"payment_count_{window}d": ("amount_paid", "count"),
                    f"late_payment_count_{window}d": ("is_late", "sum"),
                    f"days_late_avg_{window}d": ("days_late", "mean"),
                    f"days_late_trend_{window}d": ("days_late", lambda x: x.corr(pd.Series(range(len(x)))) if len(x) > 1 else 0.0),
                    f"total_late_amount_{window}d": ("amount_paid", lambda x: x.sum() if len(x) > 0 else 0.0),
                }
            ).reset_index()
            features_list.append(agg)

        if not features_list:
            return pd.DataFrame({"tenant_id": []})

        result = features_list[0]
        for f in features_list[1:]:
            result = result.merge(f, on="tenant_id", how="outer")
        return result

    def _build_maintenance_features(self, as_of: pd.Timestamp) -> pd.DataFrame:
        maint = self.tables["maintenance_requests"].copy()
        if maint.empty:
            return pd.DataFrame({"tenant_id": []})

        # Tag all maintenance text first
        tagged = maint["text"].apply(
            lambda t: pd.Series(tag_maintenance_text(t))
        )
        maint["severity"] = tagged["severity"]
        maint["sentiment"] = tagged["sentiment"]
        maint["tag_source"] = tagged["tag_source"]
        maint["event_date"] = maint["request_date"]

        features_list = []
        for window in [30, 60, 90]:
            window_events = compute_point_in_time_features(maint, as_of, window)
            if window_events.empty:
                continue

            severity_map = {"critical": 3, "moderate": 2, "minor": 1, "unknown": 0}
            window_events["severity_score"] = window_events["severity"].map(severity_map).fillna(0)

            agg = window_events.groupby("tenant_id").agg(
                **{
                    f"complaint_count_{window}d": ("severity_score", "count"),
                    f"complaint_severity_weighted_{window}d": ("severity_score", "sum"),
                    f"complaint_sentiment_avg_{window}d": ("sentiment", "mean"),
                    f"avg_resolve_days_{window}d": ("days_to_resolve", "mean"),
                }
            ).reset_index()
            features_list.append(agg)

        if not features_list:
            return pd.DataFrame({"tenant_id": []})

        result = features_list[0]
        for f in features_list[1:]:
            result = result.merge(f, on="tenant_id", how="outer")
        return result

    def _build_lease_features(self, as_of: pd.Timestamp) -> pd.DataFrame:
        leases = self.tables["leases"].copy()
        leases["event_date"] = leases["lease_start"]

        window_events = compute_point_in_time_features(leases, as_of, 730)
        if window_events.empty:
            return pd.DataFrame({"tenant_id": []})

        current_leases = window_events[
            (window_events["lease_start"] <= as_of) & (window_events["lease_end"] > as_of)
        ]
        if current_leases.empty:
            return pd.DataFrame({"tenant_id": []})

        current_leases["tenure_days"] = (as_of - current_leases["lease_start"]).dt.days
        current_leases["tenure_months"] = (current_leases["tenure_days"] / 30).astype(int)
        current_leases["lease_remaining_months"] = (
            (current_leases["lease_end"] - as_of).dt.days / 30
        ).astype(int)
        current_leases["rent_gap_pct"] = (
            current_leases["current_rent"] - current_leases["initial_rent"]
        ) / current_leases["initial_rent"].clip(lower=1)

        return current_leases[
            ["tenant_id", "tenure_months", "lease_remaining_months",
             "rent_gap_pct", "current_rent"]
        ].reset_index(drop=True)

    def _build_market_features(self, as_of: pd.Timestamp) -> pd.DataFrame:
        comps = self.tables["market_comps"].copy()
        comps["event_date"] = comps["month"]
        window_events = compute_point_in_time_features(comps, as_of, 365)
        if window_events.empty:
            return pd.DataFrame({"tenant_id": []})

        market_avg = window_events.groupby("neighborhood_cluster")["market_rent"].mean().reset_index()
        market_avg.columns = ["neighborhood_cluster", "avg_market_rent"]

        units = self.tables["units"][["unit_id", "neighborhood_cluster", "base_rent"]].copy()
        units = units.merge(market_avg, on="neighborhood_cluster", how="left")
        units["market_rent_gap"] = (units["avg_market_rent"] - units["base_rent"]) / units["base_rent"].clip(lower=1)

        leases = self.tables["leases"][["lease_id", "unit_id", "tenant_id"]]
        result = leases.merge(units, on="unit_id", how="left")
        return result[["tenant_id", "market_rent_gap", "avg_market_rent"]]

    def _build_intervention_features(self, as_of: pd.Timestamp) -> pd.DataFrame:
        iv = self.tables["intervention_log"].copy()
        if iv.empty:
            return pd.DataFrame({"tenant_id": []})
        iv["event_date"] = iv["offered_date"]

        features_list = []
        for window in [180, 365]:
            window_events = compute_point_in_time_features(iv, as_of, window)
            if window_events.empty:
                continue

            agg = window_events.groupby("tenant_id").agg(
                **{
                    f"prior_interventions_{window}d": ("intervention_type", "count"),
                    f"prior_intervention_accept_rate_{window}d": ("accepted", "mean"),
                }
            ).reset_index()
            features_list.append(agg)

        if not features_list:
            return pd.DataFrame({"tenant_id": []})

        result = features_list[0]
        for f in features_list[1:]:
            result = result.merge(f, on="tenant_id", how="outer")
        return result

    def _build_tenant_static_features(self) -> pd.DataFrame:
        tenants = self.tables["tenants"][
            ["tenant_id", "household_size", "voucher_holder", "zip_code"]
        ].copy()
        tenants["is_voucher_holder"] = tenants["voucher_holder"].astype(int)
        tenants["household_size_large"] = (tenants["household_size"] > 3).astype(int)
        return tenants.drop(columns=["voucher_holder"])

    def build(self, as_of_months: list[pd.Timestamp] | None = None) -> pd.DataFrame:
        if as_of_months is None:
            start = pd.Timestamp(self.tables["leases"]["lease_start"].min())
            end = pd.Timestamp(self.tables["leases"]["lease_end"].max())
            as_of_months = list(pd.date_range(start, end, freq="MS"))

        all_features = []
        static_features = self._build_tenant_static_features()

        for as_of in as_of_months:
            logger.debug(f"Building features as of {as_of.date()}")

            try:
                payment_feats = self._build_payment_features(as_of)
                maint_feats = self._build_maintenance_features(as_of)
                lease_feats = self._build_lease_features(as_of)
                market_feats = self._build_market_features(as_of)
                intervention_feats = self._build_intervention_features(as_of)

                merged = static_features.copy()
                merged["as_of_month"] = as_of

                for df in [payment_feats, maint_feats, lease_feats, market_feats, intervention_feats]:
                    if df.empty or "tenant_id" not in df.columns:
                        continue
                    merged = merged.merge(
                        df, on="tenant_id", how="left"
                    )

                all_features.append(merged)
            except Exception as e:
                logger.error(f"Feature build failed for as_of {as_of}: {e}")
                raise FeatureBuildError(f"Feature build failed at {as_of}: {e}") from e

        if not all_features:
            raise FeatureBuildError("No features were generated for any month")

        feature_table = pd.concat(all_features, ignore_index=True)

        # Impute missing values — documented strategy
        numeric_cols = feature_table.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            null_pct = feature_table[col].isna().mean()
            if null_pct > 0.05:
                logger.warning(
                    f"Feature '{col}' has {null_pct:.1%} missing — "
                    f"imputing with median"
                )
            if null_pct > 0:
                feature_table[col] = feature_table[col].fillna(feature_table[col].median())

        return feature_table
