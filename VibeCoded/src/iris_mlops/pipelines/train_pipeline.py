"""End-to-end training pipeline orchestration."""

from __future__ import annotations

import logging

from iris_mlops.config import AppConfig
from iris_mlops.data.loader import DataLoader, split_features_target, train_val_test_split
from iris_mlops.features.preprocess import ScalerPreprocessor
from iris_mlops.models.evaluator import ClassificationEvaluator
from iris_mlops.models.tracking import ExperimentTracker
from iris_mlops.models.trainer import ModelTrainer
from iris_mlops.utils.io import write_json
from iris_mlops.utils.seed import set_global_seed


class TrainPipeline:
    """Orchestrates data loading, preprocessing, training, and evaluation.

    All collaborators are injected via the constructor rather than
    instantiated internally, so the pipeline depends only on abstractions
    (`DataLoader`, `ExperimentTracker`) and can be tested with fakes.

    Attributes:
        config: Full application configuration.
        data_loader: Component that loads the raw dataset.
        preprocessor: Feature scaler.
        trainer: Model trainer.
        evaluator: Metrics evaluator.
        tracker: Experiment tracking backend.
        logger: Logger used to report pipeline progress.
    """

    def __init__(
        self,
        config: AppConfig,
        data_loader: DataLoader,
        preprocessor: ScalerPreprocessor,
        trainer: ModelTrainer,
        evaluator: ClassificationEvaluator,
        tracker: ExperimentTracker,
        logger: logging.Logger,
    ) -> None:
        """Initialize the pipeline with injected collaborators.

        Args:
            config: Full application configuration.
            data_loader: Component implementing the `DataLoader` protocol.
            preprocessor: Feature preprocessor.
            trainer: Model trainer.
            evaluator: Metrics evaluator.
            tracker: Component implementing the `ExperimentTracker` protocol.
            logger: Injected logger instance.
        """
        self._config = config
        self._data_loader = data_loader
        self._preprocessor = preprocessor
        self._trainer = trainer
        self._evaluator = evaluator
        self._tracker = tracker
        self._logger = logger

    def run(self, run_name: str = "run") -> dict[str, float]:
        """Execute the full training pipeline.

        Args:
            run_name: Suffix used to name the MLflow run.

        Returns:
            A dictionary of test-set metrics.
        """
        set_global_seed(self._config.model.model.params.random_state)

        frame = self._data_loader.load()
        features, target = split_features_target(frame, self._config.data.dataset.target_column)
        x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
            features,
            target,
            test_size=self._config.data.split.test_size,
            val_size=self._config.data.split.val_size,
            random_state=self._config.data.split.random_state,
            stratify=self._config.data.split.stratify,
        )
        self._logger.info(
            "Split sizes -> train: %d, val: %d, test: %d",
            len(x_train),
            len(x_val),
            len(x_test),
        )

        with self._tracker.start_run(run_name):
            x_train_scaled = self._preprocessor.fit_transform(x_train)
            x_test_scaled = self._preprocessor.transform(x_test)

            self._trainer.train(x_train_scaled, y_train)
            model_path = self._trainer.save()
            self._preprocessor.save()

            y_pred = self._trainer.model.predict(x_test_scaled)
            metrics = self._evaluator.evaluate(y_test, y_pred)

            metrics_path = self._config.metrics.metrics.output_path
            write_json(metrics, metrics_path)

            self._tracker.log_params(self._trainer.get_params())
            self._tracker.log_metrics(metrics)
            self._tracker.log_artifact(model_path)
            self._tracker.log_artifact(metrics_path)

        self._logger.info("Training pipeline complete. Metrics: %s", metrics)
        return metrics
