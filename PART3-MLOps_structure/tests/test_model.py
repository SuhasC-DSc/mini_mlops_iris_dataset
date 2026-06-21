from iris_classifier.data import IrisDataLoader
from iris_classifier.model import IrisClassifier


def test_training() -> None:
    loader = IrisDataLoader()

    x_train, x_test, y_train, y_test = (
        loader.train_test_data()
    )

    model = IrisClassifier()

    model.train(
        x_train,
        y_train,
    )

    score = model.evaluate(
        x_test,
        y_test,
    )

    assert score > 0.8