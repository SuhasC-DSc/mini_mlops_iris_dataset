"""Configuration loading utilities.

Loads TOML configuration files and environment variables (via `.env`) into
strongly-typed, immutable configuration objects using Pydantic models.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import toml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _read_toml(filename: str) -> dict[str, Any]:
    """Read a TOML file from the configs directory.

    Args:
        filename: Name of the TOML file (e.g. ``"data.toml"``).

    Returns:
        A dictionary with the parsed TOML content.

    Raises:
        FileNotFoundError: If the requested config file does not exist.
    """
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return toml.load(path)


class SplitConfig(BaseModel):
    """Train/validation/test split configuration."""

    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    shuffle: bool = True
    stratify: bool = True


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = "iris"
    source: str = "sklearn"
    target_column: str = "target"
    raw_path: Path = Path("data/raw/iris.csv")
    processed_path: Path = Path("data/processed/iris_processed.csv")


class DataConfig(BaseModel):
    """Top-level data configuration combining dataset and split settings."""

    dataset: DatasetConfig
    split: SplitConfig


class ModelParamsConfig(BaseModel):
    """Hyperparameters passed directly to the estimator."""

    n_estimators: int = 200
    max_depth: int = 5
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    n_jobs: int = -1


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    scaler: str = "standard"
    scaler_artifact_path: Path = Path("models/scaler.joblib")


class ModelSectionConfig(BaseModel):
    """Model section configuration."""

    type: str = "random_forest"
    artifact_path: Path = Path("models/model.joblib")
    params: ModelParamsConfig


class ModelConfig(BaseModel):
    """Top-level model configuration."""

    model: ModelSectionConfig
    preprocessing: PreprocessingConfig


class MLflowSectionConfig(BaseModel):
    """MLflow experiment settings."""

    experiment_name: str = "iris-classification"
    run_name_prefix: str = "iris-rf"
    registered_model_name: str = "iris-random-forest"
    autolog: bool = False


class DagsHubSectionConfig(BaseModel):
    """DagsHub integration toggles."""

    enabled: bool = True
    repo_owner_env: str = "DAGSHUB_REPO_OWNER"
    repo_name_env: str = "DAGSHUB_REPO_NAME"


class MLflowConfig(BaseModel):
    """Top-level MLflow/DagsHub configuration."""

    mlflow: MLflowSectionConfig
    dagshub: DagsHubSectionConfig


class LoggingSectionConfig(BaseModel):
    """Logging behaviour configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_dir: Path = Path("logs")
    log_file: str = "iris_mlops.log"
    console: bool = True
    file: bool = True


class LoggingConfig(BaseModel):
    """Top-level logging configuration."""

    logging: LoggingSectionConfig


class MetricsSectionConfig(BaseModel):
    """Metrics reporting configuration."""

    output_path: Path = Path("metrics/metrics.json")
    average: str = "macro"
    report: list[str] = Field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])


class MetricsConfig(BaseModel):
    """Top-level metrics configuration."""

    metrics: MetricsSectionConfig


class Secrets(BaseModel):
    """Secrets loaded from environment variables / `.env` file."""

    mlflow_tracking_uri: str | None = None
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: str | None = None
    dagshub_user_token: str | None = None
    dagshub_repo_owner: str | None = None
    dagshub_repo_name: str | None = None
    environment: str = "development"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Secrets:
        """Build a `Secrets` instance from environment variables.

        Returns:
            A populated `Secrets` instance.
        """
        load_dotenv(override=False)
        return cls(
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            mlflow_tracking_username=os.getenv("MLFLOW_TRACKING_USERNAME"),
            mlflow_tracking_password=os.getenv("MLFLOW_TRACKING_PASSWORD"),
            dagshub_user_token=os.getenv("DAGSHUB_USER_TOKEN"),
            dagshub_repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
            dagshub_repo_name=os.getenv("DAGSHUB_REPO_NAME"),
            environment=os.getenv("ENVIRONMENT", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


class AppConfig(BaseModel):
    """Aggregate application configuration.

    Combines all TOML-backed configuration sections plus environment-derived
    secrets into a single immutable object that is safe to pass around via
    dependency injection.
    """

    data: DataConfig
    model: ModelConfig
    mlflow: MLflowConfig
    logging: LoggingConfig
    metrics: MetricsConfig
    secrets: Secrets

    model_config = {"frozen": True}


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Load and cache the full application configuration.

    Reads all TOML files under ``configs/`` and merges them with secrets
    loaded from the environment / `.env` file.

    Returns:
        A cached, immutable `AppConfig` instance.
    """
    return AppConfig(
        data=DataConfig(**_read_toml("data.toml")),
        model=ModelConfig(**_read_toml("model.toml")),
        mlflow=MLflowConfig(**_read_toml("mlflow.toml")),
        logging=LoggingConfig(**_read_toml("logging.toml")),
        metrics=MetricsConfig(**_read_toml("metrics.toml")),
        secrets=Secrets.from_env(),
    )
