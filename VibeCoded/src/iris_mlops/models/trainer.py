"""Model training component."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from iris_mlops.config import ModelSectionConfig
from iris_mlops.utils.io import JoblibSerializer


class Estimator(Protocol):
    """Protocol describing the subset of scikit-learn estimator API used."""

    def fit(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.int_]) -> Estimator:
        """Fit the estimator."""
        ...

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
        """Predict labels for the given samples."""
        ...


def _build_estimator(config: ModelSectionConfig) -> ClassifierMixin:
    """Instantiate a scikit-learn classifier from configuration.

    Args:
        config: Model section configuration describing type and params.

    Returns:
        An unfitted scikit-learn classifier.

    Raises:
        ValueError: If the configured model type is not supported.
    """
    if config.type == "random_forest":
        params = config.params
        return RandomForestClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            min_samples_leaf=params.min_samples_leaf,
            random_state=params.random_state,
            n_jobs=params.n_jobs,
        )
    raise ValueError(f"Unsupported model type: {config.type}")


class ModelTrainer:
    """Trains a classifier and persists it via Joblib.

    Attributes:
        config: Model configuration (type, hyperparameters, artifact path).
        logger: Logger used to report training progress.
        serializer: Component responsible for Joblib persistence.
    """

    def __init__(
        self,
        config: ModelSectionConfig,
        logger: logging.Logger,
        serializer: JoblibSerializer | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Model configuration.
            logger: Injected logger instance.
            serializer: Optional custom serializer; defaults to
                `JoblibSerializer`.
        """
        self._config = config
        self._logger = logger
        self._serializer = serializer or JoblibSerializer()
        self._model: ClassifierMixin = _build_estimator(config)

    def train(
        self, x_train: npt.NDArray[np.float64], y_train: npt.NDArray[np.int_]
    ) -> ClassifierMixin:
        """Fit the underlying estimator.

        Args:
            x_train: Training feature matrix.
            y_train: Training target vector.

        Returns:
            The fitted scikit-learn estimator.
        """
        self._logger.info("Training %s on %d samples", self._config.type, x_train.shape[0])
        self._model.fit(x_train, y_train)
        return self._model

    def save(self, path: Path | None = None) -> Path:
        """Persist the fitted model to disk via Joblib.

        Args:
            path: Optional override for the artifact path.

        Returns:
            The path the model was saved to.
        """
        target = Path(path or self._config.artifact_path)
        self._serializer.save(self._model, target)
        self._logger.info("Saved trained model to %s", target)
        return target

    @property
    def model(self) -> ClassifierMixin:
        """The underlying fitted (or unfitted) estimator."""
        return self._model

    def get_params(self) -> dict[str, object]:
        """Return the estimator's hyperparameters for logging purposes.

        Returns:
            A dictionary of hyperparameter names to values.
        """
        return dict(self._model.get_params())
