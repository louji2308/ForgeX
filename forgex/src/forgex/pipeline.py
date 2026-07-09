"""End-to-end training pipeline for ForgeX.

Orchestrates Phases 1-7: generate synthetic data → build features →
reshape to person-period → train baseline + primary models →
compute SHAP explanations.

Usage:
    python -m forgex.pipeline              # full pipeline
    python -m forgex.pipeline --quick       # smaller dataset for testing
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forgex.config import load_settings
from forgex.errors import ForgeXError
from forgex.logging_setup import get_logger
from forgex.models.baseline import BaselineHazardModel, hazard_to_survival
from forgex.models.hazard import train_hazard_model, ModelArtifact
from forgex.models.uplift import (
    TLearnerCATE,
    check_positivity,
    fit_cate_models,
    validate_cate_recovers_segments,
    save_cate_models,
)
from forgex.simulation.generator import SyntheticWorldEngine
from forgex.features.pipeline import FeaturePipeline
from forgex.reshape.person_period import reshape_pipeline
from forgex.explain.shap_explainer import ShapExplainer
from forgex.fairness.audit import (
    run_fairness_audit,
    demonstrate_bias_correction,
)
from forgex.mlops.evaluation import generate_evaluation_report

logger = get_logger(__name__)


def save_tables(tables: dict[str, pd.DataFrame], data_dir: Path) -> None:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = raw_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Saved {name}: {len(df)} rows -> {path}")


def _check_core_deps():
    missing = []
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        missing.append("lightgbm")
    try:
        import shap  # noqa: F401
    except ImportError:
        missing.append("shap")
    try:
        import pulp  # noqa: F401
    except ImportError:
        missing.append("pulp")
    if missing:
        raise ForgeXError(
            f"Missing core dependencies: {missing}. "
            f"Install with: pip install {' '.join(missing)}"
        )


def main(quick: bool = False):
    _check_core_deps()
    settings = load_settings()

    if quick:
        settings.data.n_tenants = 500
        settings.data.n_properties = 20
        logger.info("Running in QUICK mode (500 tenants)")

    # Phase 1: Generate synthetic world
    logger.info("=" * 60)
    logger.info("Phase 1: Synthetic World Engine")
    logger.info("=" * 60)
    engine = SyntheticWorldEngine(settings)
    tables, hidden_segments, bias_ground_truth = engine.generate()
    save_tables(tables, settings.data_dir)

    hidden_dir = settings.data_dir / "raw"
    hidden_segments.to_parquet(hidden_dir / "hidden_segments.parquet", index=False)
    bias_ground_truth.to_parquet(hidden_dir / "hidden_bias_ground_truth.parquet", index=False)
    logger.info(f"Saved hidden segments ({len(hidden_segments)} rows) and bias ground truth")

    # Phase 2: Build features
    logger.info("=" * 60)
    logger.info("Phase 2: Feature Store & NLP Pipeline")
    logger.info("=" * 60)
    feature_pipeline = FeaturePipeline(tables)
    as_of_months = list(pd.date_range(
        pd.Timestamp(settings.data.start_date) + pd.DateOffset(months=1),
        pd.Timestamp(settings.data.end_date),
        freq="MS",
    ))
    feature_table = feature_pipeline.build(as_of_months=as_of_months)
    processed_dir = settings.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(processed_dir / "feature_table.parquet", index=False)
    logger.info(f"Feature table: {len(feature_table)} rows, {len(feature_table.columns)} columns")

    # Phase 3: Person-period reshape
    logger.info("=" * 60)
    logger.info("Phase 3: Person-Period Reshape")
    logger.info("=" * 60)
    as_of_date = pd.Timestamp(settings.data.end_date)
    reshape_result = reshape_pipeline(
        tables, feature_table, as_of_date=as_of_date,
        output_dir=processed_dir,
    )
    person_period = reshape_result["person_period"]

    # Phase 4: Baseline Hazard Model
    logger.info("=" * 60)
    logger.info("Phase 4: Baseline Hazard Model")
    logger.info("=" * 60)
    baseline = BaselineHazardModel()
    baseline.fit(person_period)
    logger.info(f"Baseline PR-AUC: {baseline.artifact.pr_auc:.3f}")

    baseline_survival = baseline.predict_survival_curves(
        person_period[person_period["fold"] == "test"]
    )
    baseline_survival.to_parquet(processed_dir / "baseline_survival_curves.parquet", index=False)

    # Phase 5: Primary Hazard Model (LightGBM)
    logger.info("=" * 60)
    logger.info("Phase 5: Primary Hazard Model (LightGBM)")
    logger.info("=" * 60)
    feature_cols = [c for c in baseline.artifact.feature_names
                    if c not in {"tenant_id", "lease_id", "calendar_month",
                                 "month_of_lease", "fold", "as_of_month",
                                 "is_censored", "still_active"}]

    train = person_period[person_period["fold"] == "train"]
    val = person_period[person_period["fold"] == "val"]

    X_train = train[feature_cols].fillna(0)
    y_train = train["churn_event_this_month"]
    tenant_train = train["tenant_id"]

    X_val = val[feature_cols].fillna(0)
    y_val = val["churn_event_this_month"]

    combined_X = pd.concat([X_train, X_val], ignore_index=True)
    combined_y = pd.concat([y_train, y_val], ignore_index=True)
    combined_tenants = pd.concat([tenant_train, val["tenant_id"]], ignore_index=True)

    logger.info(f"Training LightGBM on {len(combined_X)} rows, {len(feature_cols)} features")
    try:
        hazard_artifact = train_hazard_model(
            combined_X, combined_y, combined_tenants,
            categorical_features=["is_voucher_holder", "household_size_large"],
        )
        logger.info(f"LightGBM OOF PR-AUC: {hazard_artifact.pr_auc:.3f}")
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        logger.warning("Falling back to baseline model for predictions")
        hazard_artifact = ModelArtifact(
            model=baseline.artifact.model,
            feature_names=baseline.artifact.feature_names,
            categorical_features=[],
            train_event_rate=baseline.artifact.train_event_rate,
            pr_auc=baseline.artifact.pr_auc,
        )
        hazard_artifact.model._is_baseline_fallback = True

    artifacts_dir = settings.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "hazard_model.pkl", "wb") as f:
        pickle.dump(hazard_artifact, f)
    logger.info(f"Saved hazard model to {artifacts_dir / 'hazard_model.pkl'}")

    # Phase 6: SHAP Explainability
    logger.info("=" * 60)
    logger.info("Phase 6: Explainability Layer (SHAP)")
    logger.info("=" * 60)
    explainer = ShapExplainer(hazard_artifact.model, hazard_artifact.feature_names)
    explainer.fit(combined_X)
    explainer.save(artifacts_dir / "shap_explainer.pkl")

    # Compute survival curves
    logger.info("Computing survival curves...")
    test_pp = person_period[person_period["fold"] == "test"].copy()
    if not test_pp.empty:
        test_hazards = hazard_artifact.model.predict(
            test_pp[hazard_artifact.feature_names].fillna(0)
        )
        if hasattr(hazard_artifact.model, "_calibrator"):
            test_hazards = hazard_artifact.model._calibrator.predict(test_hazards)
        survival_df = hazard_to_survival(
            pd.Series(test_hazards),
            test_pp["tenant_id"],
            test_pp["month_of_lease"],
        )
        survival_df.to_parquet(artifacts_dir / "survival_curves.parquet", index=False)
        logger.info(f"Saved survival curves for {survival_df['tenant_id'].nunique()} tenants")

    # Phase 8: Causal Uplift Model
    logger.info("=" * 60)
    logger.info("Phase 8: Causal Uplift Model (CATE)")
    logger.info("=" * 60)
    intervention_log = tables.get("intervention_log")
    if intervention_log is not None and not intervention_log.empty:
        iv_log = _prepare_cate_data(
            intervention_log, tables, train, feature_cols
        )
        if iv_log is None:
            logger.warning("CATE data preparation failed — skipping Phase 8")
        else:
            common_covariates = [c for c in feature_cols if c in iv_log.columns][:8]
            if not common_covariates:
                logger.warning("No common covariates for CATE — skipping Phase 8")
            else:
                try:
                    cate_models = fit_cate_models(
                        iv_log[common_covariates + ["tenant_id", "intervention_type", "renewed"]],
                        common_covariates,
                        outcome_col="renewed",
                        intervention_col="intervention_type",
                        use_t_learner=True,
                    )
                    save_cate_models(cate_models, artifacts_dir / "cate")

                    hidden_path = hidden_dir / "hidden_segments.parquet"
                    if hidden_path.exists():
                        hidden_seg = pd.read_parquet(hidden_path)
                        train_cov = train[common_covariates].fillna(0)
                        cate_estimates = cate_models["t_learner"].predict_cate(train_cov)
                        segment_validation = validate_cate_recovers_segments(
                            cate_estimates, hidden_seg
                        )
                        logger.info(
                            f"Segment recovery validation: "
                            f"p_value={segment_validation['p_value']:.4f}, "
                            f"recovers_segmentation={segment_validation['recovers_segmentation']}"
                        )
                        if segment_validation["recovers_segmentation"]:
                            logger.info("✓ CATE model recovered the hidden segmentation!")
                        else:
                            logger.warning(
                                "CATE did NOT recover hidden segmentation at p<0.01. "
                                "Consider revisiting Phase 1 DGP effect sizes."
                            )
                except Exception as e:
                    logger.error(f"CATE training failed (non-blocking): {e}")
    else:
        logger.warning("No intervention_log available — skipping CATE (Phase 8)")

    # Phase 13: Fairness Audit
    logger.info("=" * 60)
    logger.info("Phase 13: Fairness Audit")
    logger.info("=" * 60)
    bias_path = hidden_dir / "hidden_bias_ground_truth.parquet"
    if bias_path.exists():
        bias_gt = pd.read_parquet(bias_path)
        if "legacy_risk_flag" in tables.get("leases", pd.DataFrame()).columns:
            leases_with_bias = tables["leases"]
            legacy_risk = leases_with_bias.groupby("tenant_id")["legacy_risk_flag"].first().reset_index()
            legacy_risk.columns = ["tenant_id", "legacy_risk_flag"]

            # Get model predictions for audit
            test_tenants = test_pp[["tenant_id"]].drop_duplicates()
            audit_df = test_tenants.merge(
                legacy_risk, on="tenant_id", how="left"
            )
            audit_df = audit_df.merge(
                bias_gt, on="tenant_id", how="left"
            )
            tenant_voucher = tables["tenants"][["tenant_id", "voucher_holder"]].copy()

            if not test_pp.empty and not audit_df.empty:
                test_preds = pd.DataFrame({
                    "tenant_id": test_pp["tenant_id"].values,
                    "prediction": hazard_artifact.model.predict(
                        test_pp[hazard_artifact.feature_names].fillna(0)
                    ),
                })
                test_preds["pred_binary"] = (test_preds["prediction"] >= hazard_artifact.train_event_rate).astype(int)
                test_preds = test_preds.drop_duplicates("tenant_id")

                audit_df = audit_df.merge(test_preds, on="tenant_id", how="inner")
                audit_df = audit_df.merge(
                    tables["leases"][["tenant_id", "did_renew"]].drop_duplicates("tenant_id"),
                    on="tenant_id", how="left"
                )

                if not audit_df.empty and "voucher_holder" in audit_df.columns:
                    audit_result = run_fairness_audit(
                        y_true=audit_df["did_renew"].fillna(True).astype(int),
                        y_pred_binary=audit_df["pred_binary"],
                        sensitive_feature=audit_df["voucher_holder"],
                    )
                    logger.info(
                        f"Fairness audit: DP diff={audit_result.demographic_parity_difference:.3f}, "
                        f"passed={audit_result.passed_gate}"
                    )
                    with open(artifacts_dir / "fairness_audit.json", "w") as f:
                        json.dump({
                            "demographic_parity_difference": audit_result.demographic_parity_difference,
                            "equalized_odds_difference": audit_result.equalized_odds_difference,
                            "selection_rate_by_group": audit_result.selection_rate_by_group,
                            "passed_gate": audit_result.passed_gate,
                        }, f, indent=2)

    # Evaluation Report
    logger.info("=" * 60)
    logger.info("Phase 17: Evaluation Report")
    logger.info("=" * 60)
    eval_report = generate_evaluation_report(
        person_period, hazard_artifact, feature_cols,
        output_path=artifacts_dir / "evaluation_report.json",
    )
    for key, value in eval_report["test_metrics"].items():
        logger.info(f"  {key}: {value}")

    logger.info("=" * 60)
    logger.info("Pipeline complete! All artifacts saved to:")
    logger.info(f"  - {artifacts_dir}")
    logger.info(f"  - {processed_dir}")
    logger.info(f"  - {settings.data_dir / 'raw'}")
    logger.info("=" * 60)


def _prepare_cate_data(
    intervention_log: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    train_pp: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame | None:
    """Prepare intervention log for CATE fitting by merging covariates
    and adding synthetic control rows."""
    try:
        iv_log = intervention_log.copy()
        all_tenants = tables["tenants"]["tenant_id"].unique()
        n_control = min(500, len(all_tenants))
        rng = np.random.default_rng(42)
        control_tids = rng.choice(all_tenants, size=n_control, replace=False)
        control_tids = [str(t) for t in control_tids]

        control_rows = pd.DataFrame({
            "tenant_id": control_tids,
            "intervention_type": "none",
            "cost_dollars": 0.0,
            "crew_hours": 0.0,
        })
        lease_renewals = tables["leases"][["tenant_id", "did_renew"]].drop_duplicates("tenant_id")
        control_rows = control_rows.merge(lease_renewals, on="tenant_id", how="left")
        control_rows["renewed"] = control_rows.get("did_renew", pd.Series([True] * len(control_rows))).fillna(True).astype(bool)
        control_rows = control_rows.drop(columns=["did_renew"], errors="ignore")

        iv_log = pd.concat([iv_log, control_rows], ignore_index=True)

        covs_to_merge = [c for c in feature_cols if c in train_pp.columns][:8]
        if not covs_to_merge:
            logger.warning("No feature columns available for CATE merge")
            return None

        tenant_covs = train_pp[["tenant_id"] + covs_to_merge].drop_duplicates("tenant_id")
        iv_log = iv_log.merge(tenant_covs, on="tenant_id", how="left")
        iv_log = iv_log.dropna(subset=covs_to_merge, how="all").fillna(0)
        return iv_log
    except Exception as e:
        logger.error(f"CATE data preparation error: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForgeX Training Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run with small dataset")
    args = parser.parse_args()
    try:
        main(quick=args.quick)
    except ForgeXError as e:
        logger.critical(f"Pipeline failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected pipeline failure: {e}", exc_info=True)
        sys.exit(1)
