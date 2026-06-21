"""Training entrypoint."""

from pathlib import Path

from iris_classifier.data import IrisDataLoader
from iris_classifier.model import IrisClassifier

MODEL_PATH = Path("models/iris_model.joblib")


def main() -> None:
    """Train and save model."""

    loader = IrisDataLoader()

    x_train, x_test, y_train, y_test = (
        loader.train_test_data()
    )

    classifier = IrisClassifier()

    classifier.train(
        x_train,
        y_train,
    )

    accuracy = classifier.evaluate(
        x_test,
        y_test,
    )

    classifier.save(MODEL_PATH)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()