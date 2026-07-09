"""Model evaluation utilities."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from iris_mlops.config import MetricsSectionConfig


class ClassificationEvaluator:
    """Computes standard classification metrics for a fitted model.

    Attributes:
        config: Metrics configuration describing which metrics to report.
        logger: Logger used to report evaluation results.
    """

    def __init__(self, config: MetricsSectionConfig, logger: logging.Logger) -> None:
        """Initialize the evaluator.

        Args:
            config: Metrics configuration.
            logger: Injected logger instance.
        """
        self._config = config
        self._logger = logger

    def evaluate(
        self,
        y_true: npt.NDArray[np.int_],
        y_pred: npt.NDArray[np.int_],
    ) -> dict[str, float]:
        """Compute configured classification metrics.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.

        Returns:
            A dictionary mapping metric name to its computed value.
        """
        available = {
            "accuracy": lambda: accuracy_score(y_true, y_pred),
            "precision": lambda: precision_score(
                y_true, y_pred, average=self._config.average, zero_division=0
            ),
            "recall": lambda: recall_score(
                y_true, y_pred, average=self._config.average, zero_division=0
            ),
            "f1": lambda: f1_score(y_true, y_pred, average=self._config.average, zero_division=0),
        }

        metrics: dict[str, float] = {}
        for name in self._config.report:
            if name not in available:
                self._logger.warning("Unknown metric '%s' requested; skipping", name)
                continue
            metrics[name] = float(available[name]())

        self._logger.info("Computed metrics: %s", metrics)
        return metrics
