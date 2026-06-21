# Iris Classifier

A production-oriented machine learning project built on the Iris dataset using scikit-learn.

## Highlights

* Managed with `uv` for dependency and environment management
* Configured using a single `pyproject.toml`
* Uses the `src/` layout for proper Python packaging
* Runtime and development dependencies separated
* Dependency versions locked with `uv.lock` for reproducibility
* Type hints throughout the codebase with MyPy configuration
* Code quality enforced with Ruff
* Unit tests implemented with Pytest
* Object-oriented design for data loading, training, and inference
* Model persistence using Joblib
* Package-based imports and module execution
* Structured for future containerization and deployment

## Commands

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
uv run python -m iris_classifier.train
uv run python -m iris_classifier.predict
```  
  
# Code Quality & Testing Configuration

## Ruff

Ruff is used for linting and import organization.

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "B"]
```

### Enabled Rules

| Code | Purpose |
|------|---------|
| `E` | PEP 8 style checks (pycodestyle) |
| `F` | Error detection such as unused imports and undefined names (Pyflakes) |
| `I` | Import sorting and grouping (isort) |
| `B` | Bug-prone patterns and best practices (flake8-bugbear) |

---

## MyPy

MyPy is used for static type checking.

```toml
[tool.mypy]
strict = true
python_version = "3.11"
ignore_missing_imports = true
check_untyped_defs = true
```

### Configuration

| Option | Description |
|---------|-------------|
| `strict = true` | Enables strict type checking and additional safety checks. |
| `python_version = "3.11"` | Type-check code against Python 3.11 features and syntax. |
| `ignore_missing_imports = true` | Suppresses errors for third-party packages without type stubs. |
| `check_untyped_defs = true` | Checks function bodies even when type annotations are missing. |

### Goal

These settings provide:

- Strong static type safety
- Early detection of type-related bugs
- Compatibility with Python 3.11
- Better maintainability and code reliability

---

## Pytest

Pytest is used for running automated tests.

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Configuration

| Option | Description |
|---------|-------------|
| `testpaths = ["tests"]` | Tells Pytest to look for and run tests only from the `tests/` directory. |

### Benefits

- Faster test discovery
- Consistent project structure
- Prevents accidental collection of non-test files
- Makes test execution predictable across environments