"""CLI entry point for running the inference pipeline.

Composition root for inference: wires up the `Predictor` and
`InferencePipeline` with concrete configuration and logger instances.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from iris_mlops.config import load_config
from iris_mlops.logger import get_logger
from iris_mlops.models.predictor import Predictor
from iris_mlops.pipelines.inference_pipeline import InferencePipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run inference with the trained Iris model.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to a CSV file containing feature columns (no target column).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/predictions.csv"),
        help="Path to write predictions to.",
    )
    return parser.parse_args()


def main() -> None:
    """Compose collaborators and execute the inference pipeline."""
    args = parse_args()
    config = load_config()
    logger = get_logger(__name__, config.logging.logging)

    if not args.input_csv.exists():
        logger.error("Input file not found: %s", args.input_csv)
        sys.exit(1)

    features = pd.read_csv(args.input_csv)

    predictor = Predictor(
        model_config=config.model.model,
        preprocessing_config=config.model.preprocessing,
        logger=logger,
    )
    pipeline = InferencePipeline(predictor=predictor, logger=logger)
    predictions = pipeline.run(features)

    output = features.copy()
    output["prediction"] = predictions
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    logger.info("Wrote predictions to %s", args.output_csv)


if __name__ == "__main__":
    main()
