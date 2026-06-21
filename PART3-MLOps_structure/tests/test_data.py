from iris_classifier.data import IrisDataLoader


def test_load_data() -> None:
    loader = IrisDataLoader()

    x, y = loader.load_data()

    assert len(x) == 150
    assert len(y) == 150