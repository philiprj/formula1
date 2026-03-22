.PHONY: install sync data features train eval export test test-all lint format typecheck pre-commit clean help

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
install:          ## Install all extras (incl. dev) via uv
	uv sync --extra all

sync:             ## Sync venv with lockfile only (no extras)
	uv sync

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
data:             ## Ingest raw lap data from FastF1
	uv run python scripts/01_ingest.py

features:         ## Build feature dataset from raw laps
	uv run python scripts/02_build_features.py

train:            ## Train all models
	uv run python scripts/03_train.py

eval:             ## Evaluate and produce metrics
	uv run python scripts/04_evaluate.py

export:           ## Export artefacts
	uv run python scripts/05_export.py

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
test:             ## Run fast tests (excludes slow/integration)
	uv run pytest -m "not slow and not integration"

test-all:         ## Run full test suite including slow tests
	uv run pytest

lint:             ## Check lint and formatting
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/

format:           ## Auto-fix lint issues and reformat
	uv run ruff check --fix src/ tests/ scripts/
	uv run ruff format src/ tests/ scripts/

typecheck:        ## Run mypy static type checks
	uv run mypy src/f1deg

pre-commit:       ## Run all pre-commit hooks against all files
	uv run pre-commit run --all-files

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------
clean:            ## Remove generated data and caches
	rm -rf data/raw/*.parquet data/processed/*.parquet data/models/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +

help:             ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
