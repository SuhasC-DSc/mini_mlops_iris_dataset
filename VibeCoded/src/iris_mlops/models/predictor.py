"""Model inference component."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import ClassifierMixin

from iris_mlops.config import ModelSectionConfig, PreprocessingConfig
from iris_mlops.features.preprocess import ScalerPreprocessor
from iris_mlops.utils.io import JoblibSerializer


class Predictor:
    """Loads a trained model and fitted preprocessor to serve predictions.

    Attributes:
        model_config: Model configuration (artifact path).
        preprocessing_config: Preprocessing configuration (scaler path).
        logger: Logger used to report prediction activity.
        serializer: Component responsible for Joblib loading.
    """

    def __init__(
        self,
        model_config: ModelSectionConfig,
        preprocessing_config: PreprocessingConfig,
        logger: logging.Logger,
        serializer: JoblibSerializer | None = None,
    ) -> None:
        """Initialize the predictor and eagerly load artifacts.

        Args:
            model_config: Model configuration.
            preprocessing_config: Preprocessing configuration.
            logger: Injected logger instance.
            serializer: Optional custom serializer; defaults to
                `JoblibSerializer`.
        """
        self._logger = logger
        self._serializer = serializer or JoblibSerializer()
        self._model: ClassifierMixin = self._serializer.load(Path(model_config.artifact_path))
        self._preprocessor = ScalerPreprocessor.load(
            config=preprocessing_config, logger=logger, serializer=self._serializer
        )
        self._logger.info("Loaded model and preprocessor artifacts for inference")

    def predict(self, features: pd.DataFrame) -> npt.NDArray[np.int_]:
        """Predict class labels for the given raw feature rows.

        Args:
            features: Raw (unscaled) feature DataFrame.

        Returns:
            An array of predicted integer class labels.
        """
        scaled = self._preprocessor.transform(features)
        predictions: npt.NDArray[np.int_] = self._model.predict(scaled)
        self._logger.info("Generated %d predictions", len(predictions))
        return predictions

    def predict_proba(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Predict class probabilities for the given raw feature rows.

        Args:
            features: Raw (unscaled) feature DataFrame.

        Returns:
            An array of shape `(n_samples, n_classes)` with class
            probabilities.

        Raises:
            AttributeError: If the underlying model does not support
                probability estimates.
        """
        if not hasattr(self._model, "predict_proba"):
            raise AttributeError("Underlying model does not support predict_proba().")
        scaled = self._preprocessor.transform(features)
        probabilities: npt.NDArray[np.float64] = self._model.predict_proba(scaled)
        return probabilities
