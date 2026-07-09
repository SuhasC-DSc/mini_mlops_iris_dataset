"""Tests for `iris_mlops.models.trainer`."""

from __future__ import annotations

import logging

import numpy as np

from iris_mlops.models.trainer import ModelTrainer


def test_train_fits_model(model_section_config, logger: logging.Logger) -> None:
    """Training should return a fitted estimator capable of predicting."""
    x_train = np.random.default_rng(0).normal(size=(20, 4))
    y_train = np.tile([0, 1], 10)

    trainer = ModelTrainer(config=model_section_config, logger=logger)
    model = trainer.train(x_train, y_train)
    predictions = model.predict(x_train)
    assert len(predictions) == len(y_train)


def test_save_creates_artifact(model_section_config, logger: logging.Logger) -> None:
    """Saving the trained model should write a Joblib artifact to disk."""
    x_train = np.random.default_rng(0).normal(size=(20, 4))
    y_train = np.tile([0, 1], 10)

    trainer = ModelTrainer(config=model_section_config, logger=logger)
    trainer.train(x_train, y_train)
    path = trainer.save()
    assert path.exists()
