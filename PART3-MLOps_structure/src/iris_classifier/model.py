"""Model utilities."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class IrisClassifier:
    """Wrapper around RandomForestClassifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Train model."""
        self.model.fit(x_train, y_train)

    def evaluate(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> float:
        """Evaluate model accuracy."""

        predictions = self.model.predict(x_test)

        return float(
            accuracy_score(
                y_test,
                predictions,
            )
        )

    def save(self, model_path: Path) -> None:
        """Save model."""

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.model,
            model_path,
        )

    @classmethod
    def load(cls, model_path: Path) -> "IrisClassifier":
        """Load model."""

        instance = cls()
        instance.model = joblib.load(model_path)

        return instance