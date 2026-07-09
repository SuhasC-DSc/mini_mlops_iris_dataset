"""Data loading abstractions for the Iris dataset.

Defines a `DataLoader` protocol plus a concrete `SklearnIrisLoader`
implementation, enabling dependency injection of alternative data sources
(e.g. a CSV-backed loader) without changing downstream code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from iris_mlops.config import DataConfig


class DataLoader(Protocol):
    """Protocol describing a component capable of loading a labeled dataset."""

    def load(self) -> pd.DataFrame:
        """Load the full dataset as a DataFrame.

        Returns:
            A DataFrame containing feature columns and the target column.
        """
        ...


class SklearnIrisLoader:
    """Loads the Iris dataset bundled with scikit-learn.

    Attributes:
        config: Data configuration describing target column and raw path.
        logger: Logger used to report loading progress.
    """

    def __init__(self, config: DataConfig, logger: logging.Logger) -> None:
        """Initialize the loader.

        Args:
            config: Data configuration object.
            logger: Logger instance injected by the caller.
        """
        self._config = config
        self._logger = logger

    def load(self) -> pd.DataFrame:
        """Load the Iris dataset and cache it as a raw CSV file.

        Returns:
            A DataFrame with feature columns plus the configured target
            column.
        """
        self._logger.info("Loading Iris dataset from sklearn")
        bunch = load_iris(as_frame=True)
        frame: pd.DataFrame = bunch.frame.rename(
            columns={"target": self._config.dataset.target_column}
        )

        raw_path = Path(self._config.dataset.raw_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(raw_path, index=False)
        self._logger.info("Saved raw dataset to %s (%d rows)", raw_path, len(frame))
        return frame


class CsvDataLoader:
    """Loads a dataset from a CSV file on disk.

    Provided as an alternative `DataLoader` implementation to demonstrate
    dependency injection / the Open-Closed Principle: pipelines depend on
    the `DataLoader` protocol, not on this concrete class.
    """

    def __init__(self, csv_path: Path, logger: logging.Logger) -> None:
        """Initialize the CSV loader.

        Args:
            csv_path: Path to the CSV file to load.
            logger: Logger instance injected by the caller.
        """
        self._csv_path = csv_path
        self._logger = logger

    def load(self) -> pd.DataFrame:
        """Read the dataset from CSV.

        Returns:
            The loaded DataFrame.
        """
        self._logger.info("Loading dataset from %s", self._csv_path)
        return pd.read_csv(self._csv_path)


def split_features_target(
    frame: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, npt.NDArray[np.int_]]:
    """Split a DataFrame into features and target arrays.

    Args:
        frame: Full dataset including the target column.
        target_column: Name of the target column.

    Returns:
        A tuple of `(features, target)` where `target` is a NumPy integer
        array.
    """
    features = frame.drop(columns=[target_column])
    target: npt.NDArray[np.int_] = frame[target_column].to_numpy()
    return features, target


def train_val_test_split(
    features: pd.DataFrame,
    target: npt.NDArray[np.int_],
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
]:
    """Split data into train, validation, and test sets.

    Args:
        features: Feature DataFrame.
        target: Target array.
        test_size: Fraction of data reserved for the test set.
        val_size: Fraction of the *remaining* (non-test) data reserved for
            validation.
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify splits by target class.

    Returns:
        A tuple `(X_train, X_val, X_test, y_train, y_val, y_test)`.
    """
    strat_full = target if stratify else None
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_full,
    )
    strat_train_val = y_train_val if stratify else None
    relative_val_size = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=strat_train_val,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test
