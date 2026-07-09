"""MLflow / DagsHub experiment tracking integration.

Encapsulates all MLflow and DagsHub-specific code behind a small
`ExperimentTracker` interface so that trainers depend on an abstraction
rather than on the `mlflow` SDK directly (Dependency Inversion Principle).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import mlflow

from iris_mlops.config import MLflowConfig, Secrets


class ExperimentTracker(Protocol):
    """Protocol for an experiment tracking backend."""

    @contextmanager
    def start_run(self, run_name: str) -> Iterator[None]:
        """Start a tracked run as a context manager."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the active run."""
        ...

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics for the active run."""
        ...

    def log_artifact(self, path: Path) -> None:
        """Log a local file as an artifact of the active run."""
        ...


class MLflowTracker:
    """MLflow-backed experiment tracker with optional DagsHub integration.

    Attributes:
        config: MLflow/DagsHub configuration section.
        secrets: Environment-derived secrets (tracking URI, credentials).
        logger: Logger used to report tracking activity.
    """

    def __init__(self, config: MLflowConfig, secrets: Secrets, logger: logging.Logger) -> None:
        """Initialize the tracker and configure the MLflow client.

        Args:
            config: MLflow/DagsHub configuration.
            secrets: Secrets loaded from the environment.
            logger: Injected logger instance.
        """
        self._config = config
        self._secrets = secrets
        self._logger = logger
        self._configure_backend()

    def _configure_backend(self) -> None:
        """Configure the MLflow tracking backend, optionally via DagsHub."""
        if self._config.dagshub.enabled and self._secrets.dagshub_repo_owner:
            try:
                import dagshub

                dagshub.init(  # type: ignore[attr-defined]
                    repo_owner=self._secrets.dagshub_repo_owner,
                    repo_name=self._secrets.dagshub_repo_name,
                    mlflow=True,
                )
                self._logger.info("Initialized DagsHub MLflow integration")
            except Exception:  # noqa: BLE001
                self._logger.warning(
                    "DagsHub initialization failed; falling back to MLFLOW_TRACKING_URI",
                    exc_info=True,
                )

        if self._secrets.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self._secrets.mlflow_tracking_uri)

        mlflow.set_experiment(self._config.mlflow.experiment_name)

    @contextmanager
    def start_run(self, run_name: str) -> Iterator[None]:
        """Start an MLflow run.

        Args:
            run_name: Human-readable name for the run.

        Yields:
            None. The MLflow run is active for the duration of the block.
        """
        full_name = f"{self._config.mlflow.run_name_prefix}-{run_name}"
        with mlflow.start_run(run_name=full_name):
            self._logger.info("Started MLflow run: %s", full_name)
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to the active MLflow run.

        Args:
            params: Mapping of parameter names to values.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to the active MLflow run.

        Args:
            metrics: Mapping of metric names to numeric values.
        """
        mlflow.log_metrics(metrics)

    def log_artifact(self, path: Path) -> None:
        """Log a local file as an MLflow artifact.

        Args:
            path: Path to the local file to upload.
        """
        mlflow.log_artifact(str(path))
