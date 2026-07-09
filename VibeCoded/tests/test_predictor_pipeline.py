"""Integration-style tests for predictor and inference pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from iris_mlops.features.preprocess import ScalerPreprocessor
from iris_mlops.models.predictor import Predictor
from iris_mlops.models.trainer import ModelTrainer
from iris_mlops.pipelines.inference_pipeline import InferencePipeline


def test_predictor_end_to_end(
    model_section_config, preprocessing_config, logger: logging.Logger
) -> None:
    """Training then loading via Predictor should produce valid predictions."""
    frame = pd.DataFrame(
        {
            "a": np.random.default_rng(1).normal(size=40),
            "b": np.random.default_rng(2).normal(size=40),
        }
    )
    target = np.tile([0, 1], 20)

    preprocessor = ScalerPreprocessor(config=preprocessing_config, logger=logger)
    scaled = preprocessor.fit_transform(frame)
    preprocessor.save()

    trainer = ModelTrainer(config=model_section_config, logger=logger)
    trainer.train(scaled, target)
    trainer.save()

    predictor = Predictor(
        model_config=model_section_config,
        preprocessing_config=preprocessing_config,
        logger=logger,
    )
    pipeline = InferencePipeline(predictor=predictor, logger=logger)
    predictions = pipeline.run(frame)

    assert len(predictions) == len(frame)
    assert set(np.unique(predictions)).issubset({0, 1})
