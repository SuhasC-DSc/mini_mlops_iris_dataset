# Iris MLOps

A production-style, end-to-end MLOps reference project for classifying the classic
Iris dataset. Built with a `src` layout, strict typing, dependency injection, and
full experiment tracking via MLflow + DagsHub, with data/model versioning via DVC.

## Features

- **Python 3.12** with **uv** for fast, reproducible dependency management
- **`src` layout** package (`src/iris_mlops`)
- **MLflow** experiment tracking, optionally proxied through **DagsHub**
- **DVC** for data and model versioning + pipeline reproducibility
- **Ruff** (lint + format) and **MyPy** (strict static typing)
- **Pytest** suite with coverage reporting
- **Joblib** for model/scaler persistence
- **TOML** configuration files, loaded into typed Pydantic models
- **.env**-based secrets management
- OOP design following **SOLID** principles with constructor-based
  **dependency injection** throughout (data loader, preprocessor, trainer,
  evaluator, and experiment tracker are all injected as protocols/interfaces)
- Google-style docstrings and NumPy-style typing (`numpy.typing.NDArray`) everywhere
- No global mutable state — configuration and logging are passed explicitly

## Project layout

```
iris-mlops/
├── configs/                  # TOML configuration files
│   ├── data.toml
│   ├── model.toml
│   ├── mlflow.toml
│   ├── logging.toml
│   └── metrics.toml
├── src/iris_mlops/
│   ├── config.py              # TOML + .env -> typed AppConfig
│   ├── logger.py               # Logger factory (no global logger state)
│   ├── utils/
│   │   ├── io.py                # JoblibSerializer, JSON helpers
│   │   └── seed.py              # Reproducibility helpers
│   ├── data/
│   │   └── loader.py             # DataLoader protocol + sklearn/CSV impls
│   ├── features/
│   │   └── preprocess.py         # Preprocessor protocol + scaler impl
│   ├── models/
│   │   ├── trainer.py             # ModelTrainer
│   │   ├── predictor.py           # Predictor (inference)
│   │   ├── evaluator.py           # ClassificationEvaluator
│   │   └── tracking.py            # ExperimentTracker protocol + MLflow/DagsHub impl
│   ├── pipelines/
│   │   ├── train_pipeline.py      # TrainPipeline orchestrator
│   │   └── inference_pipeline.py  # InferencePipeline orchestrator
│   └── scripts/
│       ├── train.py                # Composition root for training
│       └── predict.py              # Composition root for inference
├── scripts/                  # Thin CLI wrappers (used by DVC/Docker/Make)
│   ├── train.py
│   └── predict.py
├── tests/                    # Pytest suite (unit + integration)
├── dvc.yaml                  # DVC pipeline definition
├── params.yaml                # DVC pipeline parameters
├── .dvc/config                 # DVC remote configuration (DagsHub storage)
├── pyproject.toml             # Dependencies, Ruff, MyPy, Pytest config
├── Dockerfile
├── Makefile
└── .github/workflows/ci.yml   # GitHub Actions CI
```

## Getting started

### 1. Install dependencies

```bash
uv sync --extra dev
```

### 2. Configure secrets

Copy `.env.example` to `.env` and fill in your MLflow/DagsHub credentials:

```bash
cp .env.example .env
```

### 3. Run the training pipeline

```bash
uv run python scripts/train.py --run-name baseline
# or: make train
```

This will:
1. Load the Iris dataset from scikit-learn and cache it to `data/raw/iris.csv`
2. Split it into train/validation/test sets
3. Fit a `StandardScaler` and a `RandomForestClassifier`
4. Persist both artifacts to `models/` via Joblib
5. Evaluate on the held-out test set and write `metrics/metrics.json`
6. Log parameters, metrics, and artifacts to MLflow (via DagsHub, if configured)

### 4. Run inference

```bash
uv run python scripts/predict.py --input-csv path/to/features.csv --output-csv predictions.csv
# or: make predict CLI_ARGS="--input-csv path/to/features.csv"
```

### 5. Reproduce via DVC

```bash
dvc remote add -d dagshub s3://dvc \
  --force
dvc remote modify dagshub endpointurl https://dagshub.com/<user>/iris-mlops.s3
dvc repro
dvc push
```

## Quality gates

```bash
make lint        # ruff check .
make format       # ruff format .
make typecheck    # mypy src (strict mode)
make test         # pytest with coverage
make check        # all of the above
```

All of these also run automatically in CI (`.github/workflows/ci.yml`) on every
push and pull request to `main`.

## Docker

```bash
docker build -t iris-mlops .
docker run --env-file .env iris-mlops
```

## Design notes

- **Dependency Inversion / Injection**: `TrainPipeline` and `InferencePipeline`
  depend only on protocols (`DataLoader`, `ExperimentTracker`) — concrete
  implementations (`SklearnIrisLoader`, `MLflowTracker`) are wired together in
  the composition roots (`scripts/train.py`, `scripts/predict.py`).
- **Single Responsibility**: each class (loader, preprocessor, trainer,
  evaluator, tracker) owns exactly one concern.
- **Open/Closed**: new data sources (`CsvDataLoader`) or scalers (`minmax`)
  can be added without modifying pipeline code.
- **No global state**: configuration is loaded once via a cached factory
  function (`load_config()`) and passed explicitly; loggers are created via
  `get_logger(name, config)` rather than a module-level singleton.
