"""Shared Pytest fixtures."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iris_mlops.config import (
    DataConfig,
    DatasetConfig,
    LoggingSectionConfig,
    MetricsSectionConfig,
    ModelParamsConfig,
    ModelSectionConfig,
    PreprocessingConfig,
    SplitConfig,
)


@pytest.fixture
def logger() -> logging.Logger:
    """Provide a lightweight logger for tests."""
    return logging.getLogger("test")


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """Provide a small synthetic Iris-like DataFrame."""
    rng = np.random.default_rng(42)
    data = {
        "sepal length (cm)": rng.uniform(4, 8, 30),
        "sepal width (cm)": rng.uniform(2, 4.5, 30),
        "petal length (cm)": rng.uniform(1, 7, 30),
        "petal width (cm)": rng.uniform(0.1, 2.5, 30),
        "target": np.tile([0, 1, 2], 10),
    }
    return pd.DataFrame(data)


@pytest.fixture
def dataset_config(tmp_path: Path) -> DataConfig:
    """Provide a `DataConfig` pointing at a temp directory."""
    return DataConfig(
        dataset=DatasetConfig(
            name="iris",
            source="sklearn",
            target_column="target",
            raw_path=tmp_path / "raw.csv",
            processed_path=tmp_path / "processed.csv",
        ),
        split=SplitConfig(test_size=0.2, val_size=0.1, random_state=42, stratify=True),
    )


@pytest.fixture
def preprocessing_config(tmp_path: Path) -> PreprocessingConfig:
    """Provide a `PreprocessingConfig` pointing at a temp directory."""
    return PreprocessingConfig(scaler="standard", scaler_artifact_path=tmp_path / "scaler.joblib")


@pytest.fixture
def model_section_config(tmp_path: Path) -> ModelSectionConfig:
    """Provide a `ModelSectionConfig` pointing at a temp directory."""
    return ModelSectionConfig(
        type="random_forest",
        artifact_path=tmp_path / "model.joblib",
        params=ModelParamsConfig(n_estimators=10, max_depth=3, random_state=42, n_jobs=1),
    )


@pytest.fixture
def metrics_config() -> MetricsSectionConfig:
    """Provide a `MetricsSectionConfig`."""
    return MetricsSectionConfig(
        output_path=Path("metrics.json"), average="macro", report=["accuracy", "f1"]
    )


@pytest.fixture
def logging_config() -> LoggingSectionConfig:
    """Provide a console-only `LoggingSectionConfig`."""
    return LoggingSectionConfig(console=True, file=False)
