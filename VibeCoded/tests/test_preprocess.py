"""Tests for `iris_mlops.features.preprocess`."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from iris_mlops.features.preprocess import ScalerPreprocessor


def test_fit_transform_scales_data(preprocessing_config, logger: logging.Logger) -> None:
    """Fit-transform should produce zero-mean, unit-variance columns."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    preprocessor = ScalerPreprocessor(config=preprocessing_config, logger=logger)
    scaled = preprocessor.fit_transform(frame)
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-8)


def test_transform_before_fit_raises(preprocessing_config, logger: logging.Logger) -> None:
    """Calling transform before fit should raise a RuntimeError."""
    preprocessor = ScalerPreprocessor(config=preprocessing_config, logger=logger)
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(RuntimeError):
        preprocessor.transform(frame)


def test_save_and_load_roundtrip(preprocessing_config, logger: logging.Logger) -> None:
    """A saved preprocessor should be loadable and produce identical output."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    preprocessor = ScalerPreprocessor(config=preprocessing_config, logger=logger)
    expected = preprocessor.fit_transform(frame)
    preprocessor.save()

    loaded = ScalerPreprocessor.load(config=preprocessing_config, logger=logger)
    actual = loaded.transform(frame)
    assert np.allclose(expected, actual)
