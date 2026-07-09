.PHONY: install lint format typecheck test train predict dvc-repro clean

install: ## Install project + dev dependencies with uv
	uv sync --extra dev

lint: ## Run Ruff lint checks
	uv run ruff check .

format: ## Auto-format code with Ruff
	uv run ruff format .

typecheck: ## Run MyPy static type checking
	uv run mypy src

test: ## Run the Pytest suite with coverage
	uv run pytest

check: lint typecheck test ## Run all quality gates

train: ## Run the training pipeline
	uv run python scripts/train.py

predict: ## Run inference (requires --input-csv override via CLI_ARGS)
	uv run python scripts/predict.py $(CLI_ARGS)

dvc-repro: ## Reproduce the DVC pipeline
	uv run dvc repro

dvc-push: ## Push data/model artifacts to the DVC remote
	uv run dvc push

clean: ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
