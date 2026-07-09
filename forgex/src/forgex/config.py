from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class DataConfig(BaseSettings):
    n_tenants: int = Field(10_000, gt=0, le=200_000)
    n_properties: int = Field(120, gt=0)
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    seed: int = 42

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v, info):
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError(f"end_date {v} must be after start_date {start}")
        return v


class FairnessConfig(BaseSettings):
    max_dp_diff: float = Field(0.10, ge=0.0, le=1.0)


class OptimizerConfig(BaseSettings):
    monthly_budget: float = Field(5000.0, gt=0.0)
    monthly_crew_hours: float = Field(80.0, gt=0.0)
    solver_time_limit_s: int = Field(30, gt=0, le=300)


class DriftConfig(BaseSettings):
    psi_threshold: float = Field(0.25, ge=0.0, le=1.0)


class MlopsConfig(BaseSettings):
    min_relative_improvement: float = Field(0.02, ge=0.0, le=1.0)


class NarrativeConfig(BaseSettings):
    max_retries: int = Field(2, ge=0, le=10)
    timeout_s: float = Field(6.0, ge=1.0, le=30.0)


class Settings(BaseSettings):
    data: DataConfig = DataConfig()
    fairness: FairnessConfig = FairnessConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    drift: DriftConfig = DriftConfig()
    mlops: MlopsConfig = MlopsConfig()
    narrative: NarrativeConfig = NarrativeConfig()

    artifacts_dir: Path = Field(default=Path("artifacts"))
    data_dir: Path = Field(default=Path("data"))
    log_level: str = "INFO"

    anthropic_api_key: str | None = None
    webhook_secrets: dict[str, str] = {"mock": "dev-secret-do-not-use-in-prod"}

    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__",
        "env_prefix": "FORGEX__",
    }

    @model_validator(mode="after")
    def _ensure_dirs(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "interim").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        return self


def load_settings() -> Settings:
    try:
        settings = Settings()
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}") from e
    return settings
