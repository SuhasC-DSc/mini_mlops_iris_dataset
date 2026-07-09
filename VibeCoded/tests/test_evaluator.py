"""Tests for `iris_mlops.models.evaluator`."""

from __future__ import annotations

import logging

import numpy as np

from iris_mlops.models.evaluator import ClassificationEvaluator


def test_evaluate_perfect_predictions(metrics_config, logger: logging.Logger) -> None:
    """Perfect predictions should yield a score of 1.0 for all metrics."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()

    evaluator = ClassificationEvaluator(config=metrics_config, logger=logger)
    metrics = evaluator.evaluate(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_returns_requested_metrics_only(metrics_config, logger: logging.Logger) -> None:
    """Only metrics listed in config.report should be returned."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0])

    evaluator = ClassificationEvaluator(config=metrics_config, logger=logger)
    metrics = evaluator.evaluate(y_true, y_pred)

    assert set(metrics.keys()) == {"accuracy", "f1"}
