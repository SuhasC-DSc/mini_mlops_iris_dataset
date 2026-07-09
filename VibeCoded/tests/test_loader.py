"""Tests for `iris_mlops.data.loader`."""

from __future__ import annotations

import logging

import pandas as pd

from iris_mlops.data.loader import (
    SklearnIrisLoader,
    split_features_target,
    train_val_test_split,
)


def test_sklearn_iris_loader_returns_expected_shape(dataset_config, logger: logging.Logger) -> None:
    """The loader should return 150 rows with the configured target column."""
    loader = SklearnIrisLoader(config=dataset_config, logger=logger)
    frame = loader.load()
    assert len(frame) == 150
    assert dataset_config.dataset.target_column in frame.columns


def test_split_features_target(sample_frame: pd.DataFrame) -> None:
    """Splitting should separate features from the target column."""
    features, target = split_features_target(sample_frame, "target")
    assert "target" not in features.columns
    assert len(target) == len(sample_frame)


def test_train_val_test_split_sizes(sample_frame: pd.DataFrame) -> None:
    """The three-way split should respect approximate requested proportions."""
    features, target = split_features_target(sample_frame, "target")
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        features,
        target,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        stratify=False,
    )
    total = len(x_train) + len(x_val) + len(x_test)
    assert total == len(sample_frame)
    assert len(y_train) == len(x_train)
    assert len(y_val) == len(x_val)
    assert len(y_test) == len(x_test)
