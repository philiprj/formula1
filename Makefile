.PHONY: install sync data data-practice data-all results features train train-all train-anomaly eval tune export test test-all lint format typecheck pre-commit clean help

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

data-practice:    ## Ingest practice + qualifying sessions from FastF1
	uv run python scripts/01_ingest.py --sessions FP1,FP2,FP3,Q

data-all:         ## Ingest all session types (practice + qualifying + race)
	uv run python scripts/01_ingest.py --sessions all

results:          ## Ingest race results (retirements) from Jolpica API
	uv run python scripts/01b_ingest_results.py

features:         ## Build feature dataset from raw laps
	uv run python scripts/02_build_features.py

train:            ## Train a single model (usage: make train MODEL=gbm)
	uv run python scripts/03_train.py $(MODEL)

train-all:        ## Train all degradation models + anomaly model
	uv run python scripts/03_train.py linear
	uv run python scripts/03_train.py bayesian --svi
	uv run python scripts/03_train.py gbm
	uv run python scripts/06_train_anomaly.py

train-anomaly:    ## Train anomaly/retirement prediction model only
	uv run python scripts/06_train_anomaly.py

eval:             ## Evaluate and produce metrics (usage: make eval MODEL=gbm)
	uv run python scripts/04_evaluate.py $(MODEL)

tune:             ## Run Optuna hyperparameter tuning (usage: make tune MODEL=gbm [BACKEND=lightgbm] [TRIALS=150] [APPLY=true])
	uv run python scripts/05_tune.py $(MODEL) $(if $(BACKEND),--backend $(BACKEND)) $(if $(TRIALS),--n-trials $(TRIALS)) $(if $(FOLDS),--max-folds $(FOLDS)) $(if $(APPLY),--apply)

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
