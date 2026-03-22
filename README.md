# F1 Tire Degradation & Race Strategy

Data-driven tire degradation modeling and reinforcement learning race strategy for Formula 1, built entirely from public data.

## Project Roadmap

| Project | Goal | Status |
|---------|------|--------|
| **A: Tire Degradation Model** | Predict lap time = f(tire age, compound, fuel, weather) | In progress |
| **B: RL Pit Stop Strategy** | Gymnasium simulator + PPO agent for pit/compound decisions | Planned |
| **C: Multi-Agent 2026 Regs** | Opponent modeling, ERS management, self-play | Planned |

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the full roadmap and technical details.

## Project A: Tire Degradation Model

Four model tiers with increasing sophistication:

1. **Linear baseline** — Ridge regression
2. **Bayesian hierarchical state-space** — NumPyro, extends Cappello & Hoegh (2024) to multi-driver/multi-race with partial pooling
3. **Gradient boosted trees** — XGBoost/LightGBM with quantile regression
4. **Sequence model** — LSTM/GRU with MC dropout for uncertainty

### Quickstart

```bash
# Install (requires Python 3.11+)
uv pip install -e ".[all]"

# 1. Ingest race data (2022-2025, ~4-6 hours first run, then cached)
python scripts/01_ingest.py

# 2. Build features (~50k clean laps)
python scripts/02_build_features.py

# 3. Train a model
python scripts/03_train.py linear
python scripts/03_train.py gbm

# 4. Evaluate with leave-one-race-out CV
python scripts/04_evaluate.py linear

# 5. Export for Project B
python scripts/05_export.py linear
```

### Install Options

```bash
# Minimal (data pipeline + linear model)
uv pip install -e .

# With Bayesian modeling
uv pip install -e ".[bayesian]"

# With gradient boosted trees
uv pip install -e ".[gbm]"

# Everything
uv pip install -e ".[all]"
```

## Data Sources

- **FastF1** — Primary source: lap times, telemetry, weather (2018+)
- **Jolpica API** — Pit stop data, results (Ergast replacement, 1950-present)
- **Open-Meteo** — Historical weather fallback by circuit GPS coordinates

## Project Structure

```
├── conf/                  # YAML configuration
│   ├── base.yaml          # Seasons, circuits, fuel params
│   ├── features.yaml      # Feature engineering settings
│   └── models/            # Per-model hyperparameters
├── src/f1deg/             # Core package
│   ├── data/              # Ingestion, filtering, feature engineering
│   ├── models/            # 4 model tiers (linear, bayesian, gbm, sequence)
│   ├── eval/              # CV, metrics, reports
│   ├── viz/               # Degradation curves, diagnostics, comparison plots
│   └── export.py          # Model serialization for Project B
├── scripts/               # Pipeline entry points (01-05)
├── notebooks/             # Exploratory analysis with starter code
├── tests/                 # Pytest suite (no API calls)
└── data/                  # gitignored: cache, raw, processed, models
```

## Testing

```bash
make test
# or
pytest tests/ -v
```

## Key References

- Thomas et al., SAC '25 — PPO + XAI for pit strategy (Mercedes/Imperial)
- Cappello & Hoegh, 2025 — Bayesian state-space tire degradation (FastF1 data)
- Fieni et al., 2025/2026 — MINLP + RL benchmark, multi-agent extension
