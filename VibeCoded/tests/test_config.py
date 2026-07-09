"""Tests for `iris_mlops.config`."""

from __future__ import annotations

from iris_mlops.config import load_config


def test_load_config_returns_cached_instance() -> None:
    """load_config should return the same cached instance on repeat calls."""
    config_a = load_config()
    config_b = load_config()
    assert config_a is config_b


def test_load_config_sections_present() -> None:
    """All configuration sections should be populated."""
    config = load_config()
    assert config.data.dataset.name == "iris"
    assert config.model.model.type == "random_forest"
    assert config.mlflow.mlflow.experiment_name
    assert config.logging.logging.level
    assert config.metrics.metrics.average in {"macro", "micro", "weighted"}
