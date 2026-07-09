"""Feature preprocessing components.

Provides a scaler-backed `Preprocessor` abstraction so training and
inference share identical transformation logic, with the fitted scaler
persisted via Joblib.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from iris_mlops.config import PreprocessingConfig
from iris_mlops.utils.io import JoblibSerializer


class Preprocessor(Protocol):
    """Protocol for a fit/transform feature preprocessor."""

    def fit(self, features: pd.DataFrame) -> Preprocessor:
        """Fit the preprocessor on training features."""
        ...

    def transform(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Transform features using the fitted preprocessor."""
        ...

    def save(self, path: Path) -> None:
        """Persist the fitted preprocessor to disk."""
        ...


def _build_scaler(name: str) -> TransformerMixin:
    """Instantiate a scikit-learn scaler by name.

    Args:
        name: Scaler identifier, either ``"standard"`` or ``"minmax"``.

    Returns:
        An unfitted scikit-learn transformer.

    Raises:
        ValueError: If the scaler name is not recognized.
    """
    scalers: dict[str, TransformerMixin] = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
    }
    if name not in scalers:
        raise ValueError(f"Unsupported scaler '{name}'. Options: {list(scalers)}")
    return scalers[name]


class ScalerPreprocessor:
    """Feature preprocessor backed by a scikit-learn scaler.

    Attributes:
        config: Preprocessing configuration (scaler type, artifact path).
        logger: Logger used to report fit/transform activity.
        serializer: Component responsible for Joblib persistence.
    """

    def __init__(
        self,
        config: PreprocessingConfig,
        logger: logging.Logger,
        serializer: JoblibSerializer | None = None,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            config: Preprocessing configuration.
            logger: Injected logger instance.
            serializer: Optional custom serializer; defaults to
                `JoblibSerializer`, illustrating constructor-based
                dependency injection with a sensible default.
        """
        self._config = config
        self._logger = logger
        self._serializer = serializer or JoblibSerializer()
        self._scaler: TransformerMixin = _build_scaler(config.scaler)
        self._fitted = False

    def fit(self, features: pd.DataFrame) -> ScalerPreprocessor:
        """Fit the underlying scaler on training features.

        Args:
            features: Training feature DataFrame.

        Returns:
            Self, to allow method chaining.
        """
        self._logger.info("Fitting %s scaler on %d samples", self._config.scaler, len(features))
        self._scaler.fit(features)
        self._fitted = True
        return self

    def transform(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Transform features with the fitted scaler.

        Args:
            features: Feature DataFrame to transform.

        Returns:
            A NumPy array of scaled features.

        Raises:
            RuntimeError: If called before `fit`.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
        result: npt.NDArray[np.float64] = self._scaler.transform(features)
        return result

    def fit_transform(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Fit the scaler and transform the given features in one step.

        Args:
            features: Training feature DataFrame.

        Returns:
            A NumPy array of scaled features.
        """
        return self.fit(features).transform(features)

    def save(self, path: Path | None = None) -> None:
        """Persist the fitted scaler to disk via Joblib.

        Args:
            path: Optional override for the artifact path; defaults to the
                path configured in `PreprocessingConfig`.
        """
        target = path or self._config.scaler_artifact_path
        self._serializer.save(self._scaler, Path(target))
        self._logger.info("Saved fitted scaler to %s", target)

    @classmethod
    def load(
        cls,
        config: PreprocessingConfig,
        logger: logging.Logger,
        serializer: JoblibSerializer | None = None,
        path: Path | None = None,
    ) -> ScalerPreprocessor:
        """Load a previously fitted preprocessor from disk.

        Args:
            config: Preprocessing configuration.
            logger: Injected logger instance.
            serializer: Optional custom serializer.
            path: Optional override for the artifact path.

        Returns:
            A `ScalerPreprocessor` instance wrapping the loaded scaler.
        """
        instance = cls(config=config, logger=logger, serializer=serializer)
        target = path or config.scaler_artifact_path
        instance._scaler = instance._serializer.load(Path(target))
        instance._fitted = True
        return instance
