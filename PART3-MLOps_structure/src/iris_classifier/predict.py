"""Prediction utilities."""

from pathlib import Path

import numpy as np

from iris_classifier.model import IrisClassifier

MODEL_PATH = Path("models/iris_model.joblib")


class Predictor:
    """Model inference wrapper."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
    ) -> None:
        self.classifier = IrisClassifier.load(
            model_path
        )

    def predict(
        self,
        features: list[float],
    ) -> int:
        """Predict Iris class."""

        array = np.array(
            features,
            dtype=float,
        ).reshape(1, -1)

        prediction = self.classifier.model.predict(
            array
        )

        return int(prediction[0])


def main() -> None:
    """Example prediction."""

    predictor = Predictor()

    prediction = predictor.predict(
        [5.1, 3.5, 1.4, 0.2]
    )

    print(prediction)


if __name__ == "__main__":
    main()