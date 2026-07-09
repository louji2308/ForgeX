from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from forgex.errors import DataValidationError, MissingDependencyError
from forgex.logging_setup import get_logger

logger = get_logger(__name__)

try:
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_lightning.callbacks import EarlyStopping

    _TFT_AVAILABLE = True
except ImportError:
    _TFT_AVAILABLE = False
    TemporalFusionTransformer = None  # type: ignore[assignment]
    TimeSeriesDataSet = None  # type: ignore[assignment]
    QuantileLoss = None
    pl = None
    EarlyStopping = None


def build_tft_dataset(
    person_period: pd.DataFrame,
    max_encoder_length: int = 12,
    max_prediction_length: int = 3,
) -> object | None:
    """Reshape person-period data into pytorch-forecasting's TimeSeriesDataSet."""
    if not _TFT_AVAILABLE:
        raise MissingDependencyError(
            "pytorch-forecasting and pytorch-lightning are required for TFT. "
            "Install with: pip install pytorch-forecasting pytorch-lightning"
        )

    required = {"tenant_id", "month_of_lease", "churn_event_this_month"}
    if missing := required - set(person_period.columns):
        raise DataValidationError(
            f"person_period missing columns required by TFT: {missing}"
        )

    too_short = person_period.groupby("tenant_id").size()
    too_short = too_short[too_short < max_encoder_length // 2]
    if len(too_short):
        logger.warning(
            f"{len(too_short)} tenants have < {max_encoder_length // 2} months of "
            f"history and will be dropped by TFT windowing — expected for new "
            f"move-ins, but verify this isn't most of your dataset."
        )

    time_varying_unknown = [
        c for c in person_period.columns
        if c not in {
            "tenant_id", "lease_id", "calendar_month", "still_active",
            "is_censored", "fold", "as_of_month",
        }
        and c != "churn_event_this_month"
        and pd.api.types.is_numeric_dtype(person_period[c])
    ][:10]

    try:
        dataset = TimeSeriesDataSet(
            person_period,
            time_idx="month_of_lease",
            target="churn_event_this_month",
            group_ids=["tenant_id"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=[],
            time_varying_known_reals=["month_of_lease"],
            time_varying_unknown_reals=time_varying_unknown,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=True,
        )
        return dataset
    except Exception as e:
        raise RuntimeError(
            f"Failed to construct TimeSeriesDataSet: {e}"
        ) from e


def train_tft_with_budget(
    dataset,
    max_epochs: int = 15,
    max_minutes: int = 25,
    batch_size: int = 64,
    hidden_size: int = 32,
    attention_head_size: int = 2,
    dropout: float = 0.1,
    hidden_continuous_size: int = 16,
    learning_rate: float = 0.01,
) -> object | None:
    """Train TFT under a wall-clock budget, not just an epoch count."""
    if not _TFT_AVAILABLE:
        raise MissingDependencyError("pytorch-forecasting not available")

    started = time.monotonic()

    class WallClockStopper(pl.Callback):
        def on_train_batch_end(self, trainer, *_, **__):
            if (time.monotonic() - started) > max_minutes * 60:
                logger.warning(
                    f"TFT hit its {max_minutes}-minute budget, stopping."
                )
                trainer.should_stop = True

    train_loader = dataset.to_dataloader(train=True, batch_size=batch_size)
    val_loader = dataset.to_dataloader(train=False, batch_size=batch_size)

    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=1,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            WallClockStopper(),
        ],
        gradient_clip_val=0.1,
        accelerator="auto",
        enable_progress_bar=False,
        logger=False,
    )

    try:
        trainer.fit(tft, train_loader, val_loader)
        logger.info(
            f"TFT training completed in {time.monotonic() - started:.1f}s "
            f"({trainer.current_epoch} epochs)"
        )
        return tft
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError(
                "OOM during TFT training. Reduce batch_size or "
                "max_encoder_length — the demo does not depend on TFT succeeding."
            ) from e
        raise


def extract_tft_attention(tft, dataset, tenant_id: str) -> dict | None:
    """Extract attention weights for one tenant for the demo visual."""
    if not _TFT_AVAILABLE or tft is None:
        return None

    try:
        for idx in range(len(dataset)):
            if dataset.x[idx]["group_ids"][0] == tenant_id:
                raw_predictions = tft.predict(dataset[idx], mode="raw", return_x=True)
                attention = raw_predictions.attention.numpy()
                encoder_attention = attention.mean(axis=0)
                return {
                    "tenant_id": tenant_id,
                    "attention_weights": encoder_attention.tolist(),
                    "n_heads": int(attention.shape[0]),
                }
    except Exception as e:
        logger.warning(f"Could not extract attention for {tenant_id}: {e}")
        return None
