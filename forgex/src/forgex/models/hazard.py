from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression

from forgex.errors import (
    DataValidationError,
    FairnessGateFailure,
    FeatureBuildError,
    ModelSchemaError,
)
from forgex.logging_setup import get_logger

logger = get_logger(__name__)

PROXY_FEATURES = {"zip_code", "voucher_holder_status", "is_voucher_holder"}


@dataclass
class ModelArtifact:
    model: lgb.Booster
    feature_names: list[str]
    categorical_features: list[str]
    train_event_rate: float
    trained_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_sha: str | None = None
    pr_auc: float = 0.0
    calibration_applied: bool = False


def train_hazard_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tenant_ids_train: pd.Series,
    categorical_features: list[str] | None = None,
    n_splits: int = 5,
    early_stopping_rounds: int = 50,
    calibrate: bool = True,
) -> ModelArtifact:
    if X_train.empty:
        raise FeatureBuildError(
            "X_train is empty — did the reshape/feature join drop all rows?"
        )
    if y_train.nunique() < 2:
        raise DataValidationError(
            f"y_train has only {y_train.nunique()} class(es) — check "
            f"churn_event_this_month distribution upstream."
        )

    categorical_features = categorical_features or []
    if missing_cats := set(categorical_features) - set(X_train.columns):
        raise ModelSchemaError(
            f"categorical_features not in X_train: {missing_cats}"
        )

    # Tier-1 fairness sanity gate — non-negotiable even before Phase 13 exists
    if proxy_found := PROXY_FEATURES & set(X_train.columns):
        raise FairnessGateFailure(
            f"Protected-class proxy features present in training data: "
            f"{proxy_found}. Remove before training — see Phase 13."
        )

    event_rate = y_train.mean()
    if event_rate < 0.001:
        warnings.warn(
            f"Event rate is {event_rate:.4%} — PR-AUC will be noisy at this sparsity."
        )
    scale_pos_weight = (1 - event_rate) / max(event_rate, 1e-6)

    group_kfold = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(X_train))
    models = []
    fold = 0

    try:
        for fold, (tr_idx, val_idx) in enumerate(
            group_kfold.split(X_train, y_train, groups=tenant_ids_train)
        ):
            train_set = lgb.Dataset(
                X_train.iloc[tr_idx],
                y_train.iloc[tr_idx],
                categorical_feature=categorical_features,
                free_raw_data=False,
            )
            val_set = lgb.Dataset(
                X_train.iloc[val_idx],
                y_train.iloc[val_idx],
                categorical_feature=categorical_features,
                reference=train_set,
                free_raw_data=False,
            )

            booster = lgb.train(
                params={
                    "objective": "binary",
                    "metric": "average_precision",
                    "scale_pos_weight": scale_pos_weight,
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "min_data_in_leaf": 20,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "verbosity": -1,
                    "seed": 42,
                },
                train_set=train_set,
                valid_sets=[val_set],
                num_boost_round=2000,
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                ],
            )
            oof_preds[val_idx] = booster.predict(X_train.iloc[val_idx])
            models.append(booster)
    except lgb.basic.LightGBMError as e:
        raise RuntimeError(
            f"LightGBM training failed on fold {fold}: {e}. Common causes: "
            f"unencoded categorical dtype, unexpected NaNs, or a single-class fold."
        ) from e

    oof_pr_auc = average_precision_score(y_train, oof_preds)
    if oof_pr_auc < event_rate * 1.5:
        warnings.warn(
            f"OOF PR-AUC ({oof_pr_auc:.3f}) barely beats the no-skill baseline "
            f"({event_rate:.3f}) — check the feature join before tuning further."
        )

    artifact = ModelArtifact(
        model=models[-1],
        feature_names=list(X_train.columns),
        categorical_features=categorical_features,
        train_event_rate=float(event_rate),
        pr_auc=float(oof_pr_auc),
        git_sha=os.environ.get("GIT_SHA"),
    )

    if calibrate and oof_pr_auc > event_rate:
        logger.info("Applying isotonic calibration to hazard predictions...")
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(oof_preds, y_train)
        calibrated = iso_reg.predict(oof_preds)
        cal_pr_auc = average_precision_score(y_train, calibrated)
        logger.info(
            f"Calibration: PR-AUC before={oof_pr_auc:.3f}, after={cal_pr_auc:.3f}"
        )
        artifact.calibration_applied = True
        artifact.model._calibrator = iso_reg  # type: ignore[attr-defined]

    return artifact


def load_model_artifact(
    path: str, expected_features: list[str] | None = None
) -> ModelArtifact:
    try:
        with open(path, "rb") as f:
            artifact: ModelArtifact = pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        raise ModelSchemaError(
            f"Could not load model artifact from {path}: {e}"
        ) from e

    if expected_features is not None:
        missing = set(expected_features) - set(artifact.feature_names)
        extra = set(artifact.feature_names) - set(expected_features)
        if missing or extra:
            raise ModelSchemaError(
                f"Feature schema mismatch loading {path}.\n"
                f"  missing at inference time: {missing}\n"
                f"  unexpected: {extra}\n"
                f"This almost always means the feature pipeline changed after "
                f"training — retrain, don't patch around it."
            )
    return artifact
