"""Data loading utilities."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class IrisDataLoader:
    """Loads and prepares the Iris dataset."""

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load Iris dataset."""
        dataset = load_iris(as_frame=True)

        features = dataset.data
        target = dataset.target

        return features, target

    def train_test_data(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        """Return train-test split."""

        x, y = self.load_data()

        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )