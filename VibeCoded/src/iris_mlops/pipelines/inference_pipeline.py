"""End-to-end inference pipeline orchestration."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from iris_mlops.models.predictor import Predictor


class InferencePipeline:
    """Orchestrates loading input data and generating predictions.

    Attributes:
        predictor: Component that loads model/preprocessor artifacts and
            serves predictions.
        logger: Logger used to report pipeline progress.
    """

    def __init__(self, predictor: Predictor, logger: logging.Logger) -> None:
        """Initialize the pipeline.

        Args:
            predictor: Injected `Predictor` instance.
            logger: Injected logger instance.
        """
        self._predictor = predictor
        self._logger = logger

    def run(self, features: pd.DataFrame) -> npt.NDArray[np.int_]:
        """Run inference on a batch of raw feature rows.

        Args:
            features: Raw (unscaled) feature DataFrame.

        Returns:
            An array of predicted integer class labels.
        """
        self._logger.info("Running inference on %d rows", len(features))
        predictions = self._predictor.predict(features)
        self._logger.info("Inference complete")
        return predictions
