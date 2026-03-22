# F1 Tire Degradation & Race Strategy

Data-driven tire degradation modeling for Formula 1, built entirely from public data (FastF1, 2022-2025 seasons, ~165k clean race laps).

**[Live Demo](https://f1-tyre-degradation.streamlit.app/)** — interactive dashboard for exploring degradation curves, model predictions, and strategy scenarios.

## Roadmap

| Project | Goal | Status |
|---------|------|--------|
| **A: Tire Degradation** | Predict lap time from tire age, compound, fuel, weather | **Active** |
| **B: RL Strategy Agent** | Gymnasium race sim + PPO pit stop optimization | Planned |
| **C: Multi-Agent 2026** | Opponent modeling, ERS management, self-play | Planned |

Full technical plan: [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)

## Models

| Tier | Approach | MAE | Notes |
|------|----------|-----|-------|
| Linear | Ridge regression | 1.54s | Baseline |
| Bayesian | Hierarchical state-space (NumPyro) | 1.33s | Uncertainty estimates, partial pooling across circuits/drivers |
| GBM | LightGBM quantile regression | 0.77s | Best point accuracy |
| Sequence | LSTM + MC dropout | — | Needs debugging |

## Quickstart

```bash
# Install (Python 3.11+, uv recommended)
uv sync --extra all

# Pipeline
make ingest     # Pull 92 races from FastF1 (cached after first run)
make features   # Clean + feature engineer -> 165k laps
make train-all  # Train all model tiers
make evaluate   # Leave-one-race-out CV
```

### Install extras

```bash
uv sync                  # Core only (data pipeline + linear)
uv sync --extra bayesian # + NumPyro/JAX
uv sync --extra gbm      # + LightGBM/XGBoost
uv sync --extra all      # Everything
```

## Data

All free, no API keys required:

- **FastF1** — lap times, tire compounds, weather, telemetry (2018+)
- **Jolpica API** — pit stops, results (Ergast replacement)
- **Open-Meteo** — historical weather fallback

## Structure

```
conf/                  YAML configs (base, features, per-model)
src/f1deg/
  data/                Ingestion, filtering, feature engineering, schemas
  models/              Linear, Bayesian, GBM, Sequence
  eval/                Cross-validation, metrics, reports
  viz/                 Theme, degradation plots, diagnostics
scripts/               Pipeline steps (01_ingest .. 05_export)
notebooks/             EDA and model comparison
tests/                 Pytest suite (offline, no API calls)
```

## References

- Thomas et al., SAC '25 — PPO + XAI for pit strategy (Mercedes/Imperial)
- Cappello & Hoegh, 2025 — Bayesian state-space tire degradation
- Fieni et al., 2025/2026 — MINLP + RL, multi-agent extension
