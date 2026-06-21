from pathlib import Path

from iris_classifier.data import IrisDataLoader
from iris_classifier.model import IrisClassifier
from iris_classifier.predict import Predictor


def test_prediction(tmp_path: Path) -> None:
    loader = IrisDataLoader()

    x_train, _, y_train, _ = (
        loader.train_test_data()
    )

    model = IrisClassifier()

    model.train(
        x_train,
        y_train,
    )

    model_path = tmp_path / "model.joblib"

    model.save(model_path)

    predictor = Predictor(
        model_path=model_path
    )

    prediction = predictor.predict(
        [5.1, 3.5, 1.4, 0.2]
    )

    assert prediction in [0, 1, 2]