"""CLI entry point for running the training pipeline.

This module acts as the composition root: it wires together concrete
implementations of each protocol/interface and injects them into the
`TrainPipeline`. No other module in the codebase should construct these
concrete collaborators directly.
"""

from __future__ import annotations

import argparse

from iris_mlops.config import load_config
from iris_mlops.data.loader import SklearnIrisLoader
from iris_mlops.features.preprocess import ScalerPreprocessor
from iris_mlops.logger import get_logger
from iris_mlops.models.evaluator import ClassificationEvaluator
from iris_mlops.models.tracking import MLflowTracker
from iris_mlops.models.trainer import ModelTrainer
from iris_mlops.pipelines.train_pipeline import TrainPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train the Iris classification model.")
    parser.add_argument(
        "--run-name", type=str, default="baseline", help="Suffix for the MLflow run name."
    )
    return parser.parse_args()


def main() -> None:
    """Compose collaborators and execute the training pipeline."""
    args = parse_args()
    config = load_config()
    logger = get_logger(__name__, config.logging.logging)

    pipeline = TrainPipeline(
        config=config,
        data_loader=SklearnIrisLoader(config=config.data, logger=logger),
        preprocessor=ScalerPreprocessor(config=config.model.preprocessing, logger=logger),
        trainer=ModelTrainer(config=config.model.model, logger=logger),
        evaluator=ClassificationEvaluator(config=config.metrics.metrics, logger=logger),
        tracker=MLflowTracker(config=config.mlflow, secrets=config.secrets, logger=logger),
        logger=logger,
    )
    metrics = pipeline.run(run_name=args.run_name)
    logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
