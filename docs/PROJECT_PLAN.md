# F1 RL Race Strategy & Tire Degradation Modeling

## 1. Overview

This project develops a competitive reinforcement learning (RL) race strategy system for Formula 1, built entirely from publicly available data. At its core is a data-driven tire degradation model that predicts lap-time evolution as a function of tire age, compound selection, fuel load, track characteristics, and weather conditions. The tire model feeds into a Gymnasium-compliant race simulator used to train an RL agent capable of making real-time pit stop decisions that rival — and ideally surpass — the strategic calls made by F1 teams.

The work is structured as three progressively ambitious projects:

- **Project A** builds the tire degradation model from FastF1 telemetry data.
- **Project B** constructs a race simulator environment and trains an RL pit stop strategy agent.
- **Project C** extends the system to multi-agent competition with 2026 regulation changes, including ERS/battery management and opponent modeling under partial observability.

Each project is self-contained and produces publishable artifacts, but they compose naturally into a single end-to-end system.

---

## 2. Project A: Tire Degradation Model (4-6 weeks)

### Goal

Build a data-driven tire degradation model from FastF1 public data that predicts lap time as a function of tire age, compound, fuel load, track, and weather:

```
lap_time = f(tire_age, compound, fuel_load, track, weather)
```

### Data Pipeline

The primary data source is the FastF1 Python library, which provides access to lap-level and telemetry data from the 2018 season onward. The pipeline targets the 2022-2025 seasons (post-ground-effect regulation era) for consistency.

| Step | Description | Output |
|------|-------------|--------|
| **Ingest** | Pull all race sessions (2022-2025) via `fastf1.get_session()` | Raw lap DataFrames |
| **Filter** | Remove laps under Safety Car (SC), Virtual Safety Car (VSC), pit in/out laps, formation laps, and laps with obvious telemetry anomalies | Cleaned lap set |
| **Outlier removal** | Drop laps > 107% of session median or with sector-time z-scores > 3.0 | ~50,000 clean laps |
| **Enrichment** | Join weather data, driver/team metadata, circuit characteristics | Feature-complete dataset |
| **Storage** | Parquet files partitioned by season/circuit | Reproducible dataset |

### Feature Engineering

| Feature | Source | Description |
|---------|--------|-------------|
| `TyreLife` | FastF1 | Integer lap count on current tire set |
| `Compound` | FastF1 | Categorical: SOFT, MEDIUM, HARD (one-hot or ordinal) |
| `FuelLoad` | Estimated | `110 - (1.5 * current_lap)` kg; capped at 0 |
| `TrackTemp` | FastF1 weather | Track surface temperature (C) |
| `AirTemp` | FastF1 weather | Ambient air temperature (C) |
| `Humidity` | FastF1 weather | Relative humidity (%) |
| `Rainfall` | FastF1 weather | Binary rainfall indicator |
| `CircuitID` | Metadata | Categorical encoding of circuit |
| `DriverID` | Metadata | Categorical encoding of driver (for driver-skill effects) |
| `TrackPosition` | FastF1 | Position on track (proxy for dirty air / DRS effects) |
| `LapNumber` | FastF1 | Absolute lap in race (for fuel-corrected analysis) |

### Model Tiers

Four model architectures are developed in order of complexity, each serving as a benchmark for the next.

#### Tier 1: Linear Baseline (Ridge Regression)

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# Polynomial features on TyreLife to capture nonlinear degradation
poly = PolynomialFeatures(degree=3, interaction_only=False)
model = Ridge(alpha=1.0)
```

Purpose: Establish a performance floor and interpretable coefficient analysis. Expected MAE: 0.8-1.2s.

#### Tier 2: Bayesian Hierarchical State-Space Model (NumPyro)

This is the primary novel contribution. The model extends the single-driver, single-race Bayesian tire degradation model of Cappello & Hoegh (arXiv:2512.00640) to a **multi-driver, multi-race hierarchical formulation with partial pooling**.

```
Level 1 (Global):     mu_compound ~ Normal(0, sigma_global)
Level 2 (Circuit):    mu_circuit  ~ Normal(mu_compound, sigma_circuit)
Level 3 (Driver):     mu_driver   ~ Normal(mu_circuit, sigma_driver)
Level 4 (Observation): lap_time_i ~ Normal(mu_driver + f(tire_age, fuel, ...), sigma_obs)
```

Key design decisions:
- **Partial pooling** across circuits and compounds allows information sharing (e.g., a soft tire's cliff behavior at Bahrain informs expectations at Jeddah).
- **State-space formulation** models tire degradation as a latent process with Gaussian process time-varying coefficients.
- **Inference** via NUTS (No U-Turn Sampler) in NumPyro for efficient GPU-accelerated HMC.

Expected MAE: 0.4-0.7s with well-calibrated 95% prediction intervals.

#### Tier 3: Gradient Boosted Trees (XGBoost/LightGBM)

```python
import lightgbm as lgb

params = {
    "objective": "quantile",
    "alpha": 0.5,           # median regression
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "min_child_samples": 20,
}
```

Quantile regression at alpha = {0.025, 0.5, 0.975} provides prediction intervals without distributional assumptions. Feature importance via SHAP provides interpretability.

Expected MAE: 0.3-0.5s.

#### Tier 4: Sequence Model (LSTM/GRU with MC Dropout)

```python
import torch.nn as nn

class TireDegLSTM(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)
```

MC Dropout at inference time (50 forward passes) provides epistemic uncertainty estimates. Trained on stint-level sequences (variable length, padded/packed).

Expected MAE: 0.25-0.45s.

### Evaluation Protocol

| Metric | Description | Target |
|--------|-------------|--------|
| **MAE** | Mean absolute error on held-out laps | < 0.5s (Tier 3/4) |
| **95% PI Coverage** | Fraction of true lap times within predicted 95% interval | 0.93-0.97 |
| **Compound Ranking Accuracy** | Does the model correctly rank S < M < H degradation rates? | > 0.95 |
| **Cross-Validation** | Leave-one-race-out CV (train on N-1 races, test on Nth) | Stable across circuits |

### Novelty

The principal contribution is the **multi-driver, multi-race Bayesian hierarchical model with partial pooling across circuits and compounds**, built exclusively from public data. While Cappello & Hoegh demonstrated elegant single-driver state-space modeling, their approach does not pool information across the grid or across venues. The hierarchical extension enables:

1. Predictions for driver-circuit combinations not yet observed (via shrinkage toward group means).
2. Principled uncertainty quantification that decomposes into driver, circuit, compound, and observation-level variance.
3. A generative model that can be sampled forward for simulator integration (Project B).

---

## 3. Project B: RL Pit Stop Strategy Agent (8-12 weeks)

**Depends on:** Project A (tire degradation model).

### Phase 1: Race Simulator (3-4 weeks)

A Gymnasium-compliant F1 race simulator that steps lap-by-lap and supports vectorized parallel environments.

#### State Space (~12-15 dimensions)

| Index | Feature | Type | Range | Description |
|-------|---------|------|-------|-------------|
| 0 | `current_lap` | int | [1, total_laps] | Current lap number |
| 1 | `total_laps` | int | [44, 78] | Total race laps (circuit-dependent) |
| 2 | `tire_age` | int | [0, 60+] | Laps on current tire set |
| 3-5 | `tire_compound` | one-hot | {0,1}^3 | Soft / Medium / Hard |
| 6 | `estimated_tire_pace_delta` | float | [-2.0, 8.0] s | Predicted delta from tire degradation model |
| 7 | `fuel_load` | float | [0, 110] kg | Remaining fuel (decreases ~1.5 kg/lap) |
| 8 | `position` | int | [1, 20] | Current race position |
| 9 | `gap_ahead` | float | [0, 120+] s | Time gap to car ahead (inf if leading) |
| 10 | `gap_behind` | float | [0, 120+] s | Time gap to car behind (inf if last) |
| 11 | `track_temp` | float | [15, 65] C | Track surface temperature |
| 12 | `rainfall_prob` | float | [0, 1] | Probability of rain in next 5 laps |
| 13 | `safety_car_active` | bool | {0, 1} | Whether SC/VSC is currently deployed |
| 14 | `pit_stops_made` | int | [0, 5] | Number of pit stops completed |
| 15 | `compounds_used` | bitmask | [0, 7] | Bitmask of compounds used (S=1, M=2, H=4) |

```python
self.observation_space = gymnasium.spaces.Box(
    low=np.array([1, 44, 0, 0, 0, 0, -2.0, 0, 1, 0, 0, 15, 0, 0, 0, 0]),
    high=np.array([78, 78, 60, 1, 1, 1, 8.0, 110, 20, 120, 120, 65, 1, 1, 5, 7]),
    dtype=np.float32,
)
```

#### Action Space

```python
self.action_space = gymnasium.spaces.Discrete(4)
# 0: Continue (stay out)
# 1: Pit for Soft
# 2: Pit for Medium
# 3: Pit for Hard
```

#### Step Dynamics

Each `env.step(action)` advances the simulation by one lap:

1. **Lap time calculation:**
   ```
   lap_time = base_lap_time
              + tire_deg_model.predict(tire_age, compound, fuel_load, track, weather)
              + np.random.normal(0, sigma_noise)
              - fuel_effect * (110 - fuel_load) * 0.035  # ~0.035s per kg lighter
   ```

2. **Pit stop handling:** If action > 0 (pit), add circuit-specific pit loss (22-25s stationary + pit lane delta). Update tire compound and reset tire age to 0.

3. **Position updates:** Compute relative lap times for all agents (opponent strategies are pre-scripted or rule-based in Phase 1). Update positions based on cumulative time deltas. Undercuts/overcuts emerge naturally from the timing model.

4. **Safety Car model:** Each lap has a 3-5% probability of triggering a Safety Car (calibrated from historical data). SC lasts 3-6 laps. All gaps compress to ~1s under SC. This creates the "free pit stop" dynamic that is critical for realistic strategy.

5. **Regulation enforcement:** The FIA mandates use of at least 2 different dry-weather compounds per race. The environment tracks `compounds_used` and enforces this at race end.

#### Reward Structure

```python
def _compute_reward(self, action):
    reward = 1.0  # survival reward per lap

    if self.race_finished:
        # F1 points system: [25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0, ...]
        points = F1_POINTS.get(self.final_position, 0)
        reward += 100.0 * points

    if self.regulation_violation:
        reward -= 10.0  # penalty for compound rule violation

    return reward
```

### Phase 2: RL Training (2-3 weeks)

#### Training Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Environment | `gymnasium` | Env interface |
| RL algorithms | `stable-baselines3`, `sb3-contrib` | PPO, RecurrentPPO, QRDQN |
| Hyperparameter tuning | `optuna` | Bayesian optimization |
| Logging | `tensorboard`, `wandb` | Training metrics |
| Vectorization | `stable-baselines3.common.vec_env` | Parallel envs |

#### Primary Algorithm: PPO

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

env = SubprocVecEnv([make_env(seed=i) for i in range(8)])

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.999,          # long horizon (full race)
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./logs/ppo_f1/",
)

model.learn(total_timesteps=2_000_000)
```

#### Curriculum Learning

| Phase | Environment Configuration | Timesteps |
|-------|--------------------------|-----------|
| 1 | Single circuit (Bahrain), fixed weather, no SC | 500K |
| 2 | Single circuit, variable weather, SC enabled | 500K |
| 3 | Multi-circuit (5 tracks), full stochasticity | 1M |
| 4 | Full calendar (20+ tracks), all dynamics | 2-3M |

#### Hyperparameter Sweep (Optuna)

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    gamma = trial.suggest_float("gamma", 0.99, 0.9999, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    net_arch_size = trial.suggest_categorical("net_arch_size", [64, 128, 256])

    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr, n_steps=n_steps, gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=[net_arch_size, net_arch_size]),
    )
    model.learn(total_timesteps=500_000)
    return evaluate_policy(model, eval_env, n_eval_episodes=50)[0]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### Phase 3: Explainability & Validation (2-3 weeks)

#### SHAP Feature Importance

Extract feature importance from the trained policy network to understand which state features most influence pit stop decisions.

```python
import shap

explainer = shap.KernelExplainer(
    lambda obs: model.policy.predict(obs, deterministic=True)[0],
    background_data,
)
shap_values = explainer.shap_values(test_observations)
shap.summary_plot(shap_values, test_observations, feature_names=STATE_FEATURES)
```

#### VIPER: Decision Tree Distillation

Train a decision tree surrogate that approximates the neural policy, producing human-readable rules:

```python
from sklearn.tree import DecisionTreeClassifier, export_text

# Collect (observation, action) pairs from trained policy
observations, actions = collect_rollout_data(model, env, n_episodes=1000)

# Fit interpretable surrogate
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50)
dt.fit(observations, actions)
print(f"Fidelity to neural oracle: {dt.score(observations, actions):.3f}")
# Expected: ~0.97

print(export_text(dt, feature_names=STATE_FEATURES))
```

Expected output (illustrative):
```
|--- tire_age <= 18.5
|   |--- safety_car_active <= 0.5
|   |   |--- class: CONTINUE
|   |--- safety_car_active > 0.5
|   |   |--- pit_stops_made <= 0.5
|   |   |   |--- class: PIT_MEDIUM
|   |   |--- pit_stops_made > 0.5
|   |   |   |--- class: PIT_HARD
|--- tire_age > 18.5
|   |--- compound_soft <= 0.5
|   |   |--- gap_behind <= 3.5
|   |   |   |--- class: CONTINUE
|   |   |--- gap_behind > 3.5
|   |   |   |--- class: PIT_SOFT
```

#### Counterfactual Explanations

For any given race state, generate "what-if" scenarios:
- "What if the agent had pitted one lap earlier?"
- "What if it had chosen Mediums instead of Hards?"
- "What if a Safety Car had appeared on lap 30?"

These are generated by modifying single state features and re-running the policy + simulator forward.

#### Historical Race Replay

Replay real race data through the simulator and compare the agent's decisions against actual team strategies. Key comparison metrics:

- Did the agent pit on the same lap (within +/- 2 laps)?
- Did the agent choose the same compound sequence?
- Did the agent outperform the actual result (measured by cumulative race time)?

---

## 4. Project C: Multi-Agent Strategy with 2026 Regulations (12-16 weeks)

**Depends on:** Project B (trained single-agent RL system).

### 2026 Regulation Changes

The 2026 F1 technical regulations introduce significant powertrain changes that fundamentally alter strategy:

| Change | Detail | Strategy Impact |
|--------|--------|-----------------|
| **50/50 Power Split** | 50% ICE / 50% electric (up from ~20% electric) | Battery management becomes a primary strategy lever |
| **Override Mode** | Proximity-activated power boost (within 1s of car ahead) | Creates attack/defend dynamics tied to battery state |
| **Increased Battery Capacity** | ~350 kW MGU-K (up from ~120 kW) | Longer electric-only phases, more deployment strategy |
| **Active Aero** | Adjustable front and rear wing elements | Additional tactical dimension (not modeled in v1) |

### Opponent Modeling

Following Fieni et al. (arXiv:2602.23056), opponent behavior is modeled via interaction modules that capture:
- Opponent pit stop tendency as a function of tire age and gap
- Historical strategy patterns per team (e.g., Red Bull typically extends stints, Mercedes tends toward undercuts)
- Reactive strategies (opponents adjust plans based on the ego agent's actions)

### Extended State Space

The observation space grows to include battery/ERS state:

| Additional Features | Type | Range | Description |
|--------------------|------|-------|-------------|
| `battery_soc` | float | [0, 1] | Battery state of charge |
| `override_mode_available` | bool | {0, 1} | Whether Override Mode can be activated |
| `override_mode_active` | bool | {0, 1} | Whether Override Mode is currently active |
| `ers_deploy_mode` | int | [0, 4] | ERS deployment level (harvest to full deploy) |
| `opponent_tire_age_est` | float[] | [0, 60] | Estimated tire age of nearby opponents |
| `opponent_battery_est` | float[] | [0, 1] | Estimated battery SOC of nearby opponents |

### Extended Action Space

```python
self.action_space = gymnasium.spaces.MultiDiscrete([
    4,  # Tire: Continue / Pit Soft / Pit Medium / Pit Hard
    5,  # ERS: Harvest / Low / Medium / High / Overtake
    2,  # Override Mode: Off / On (if available)
])
```

### POMDP Formulation

In reality, a team cannot observe opponents' tire wear, fuel load, or battery state directly. The agent only observes:
- Positions and gaps (from timing screens)
- Sector times (from which tire performance can be inferred)
- Pit stop history (publicly visible)

Opponent internal states (tire degradation rate, battery SOC, planned strategy) are **hidden**. This naturally leads to a Partially Observable Markov Decision Process (POMDP).

### HMM-POMDP Framework

Following Kleisarchaki (arXiv:2603.01290), a Hidden Markov Model is used to maintain a belief state over opponent battery and tire conditions:

```
Belief update:
  b'(s') = eta * O(o | s', a) * sum_s T(s' | s, a) * b(s)

Where:
  s  = opponent hidden state (tire condition, battery SOC)
  o  = observed sector times, gaps, pit events
  a  = ego agent action
  T  = transition model (how opponent state evolves)
  O  = observation model (how hidden state generates observables)
```

The belief state is maintained as a particle filter (100-500 particles) and fed as additional input to the RL policy.

### Self-Play Training

The multi-agent environment supports self-play:
1. Initialize N copies of the policy (one per car on grid).
2. All agents act simultaneously each lap.
3. Train against a mixture of past policy checkpoints (to prevent forgetting).
4. League-based evaluation: Elo ratings across policy generations.

---

## 5. Data Sources

| Source | Data Available | Coverage | Access Method | License |
|--------|---------------|----------|---------------|---------|
| **FastF1** | Lap times, telemetry (30 Hz car data: speed, throttle, brake, gear, RPM, DRS), weather, tire compounds, pit stops, sector times | 2018-present | Python library (`fastf1`) | MIT |
| **OpenF1** | Real-time and historical: 3.7 Hz car telemetry, positions, stints, team radio transcriptions, race control messages | 2023-present | REST API (`api.openf1.org`) | Open |
| **Jolpica API** | Race results, qualifying, sprint, pit stops, lap times, driver/constructor standings, circuit info | 1950-present | REST API (Ergast-compatible) | CC-BY-4.0 |
| **Open-Meteo** | Historical weather by GPS coordinates: temperature, humidity, precipitation, wind | Global, 1940+ | REST API (`archive-api.open-meteo.com`) | CC-BY-4.0 |
| **F1DB (GitHub)** | Comprehensive SQLite/CSV database: all results, qualifying, standings, circuits, drivers, constructors | 1950-present | GitHub download | CC-BY-4.0 |
| **Bonomi et al. tire data** | Pirelli-collaboration tire wear model parameters, degradation curves by compound | Selected races | GitHub repository | Academic |

### Data Volume Estimates

| Dataset | Approximate Size | Records |
|---------|-----------------|---------|
| Lap data (2022-2025) | ~200 MB | ~80,000 raw laps |
| Telemetry (2022-2025) | ~15 GB | ~10B data points at 30 Hz |
| Weather (all circuits, 2022-2025) | ~50 MB | ~500K readings |
| Cleaned modeling dataset | ~100 MB | ~50,000 laps |

---

## 6. Tech Stack

### Data & ETL

| Package | Version | Purpose |
|---------|---------|---------|
| `fastf1` | >= 3.4 | F1 data access and caching |
| `pandas` | >= 2.2 | Data manipulation |
| `polars` | >= 1.0 | High-performance DataFrame operations |
| `pyarrow` | >= 17.0 | Parquet I/O |
| `requests` | >= 2.32 | API calls (OpenF1, Jolpica, Open-Meteo) |
| `SQLAlchemy` | >= 2.0 | Database access (F1DB) |

### Modeling & Statistics

| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | >= 1.5 | Ridge regression, decision trees, preprocessing |
| `xgboost` | >= 2.1 | Gradient boosted trees |
| `lightgbm` | >= 4.5 | Gradient boosted trees (alternative) |
| `numpyro` | >= 0.16 | Bayesian hierarchical models (JAX backend) |
| `jax` | >= 0.4.35 | GPU-accelerated numerical computing |
| `torch` | >= 2.5 | LSTM/GRU models, MC Dropout |
| `arviz` | >= 0.20 | Bayesian model diagnostics and visualization |

### Reinforcement Learning

| Package | Version | Purpose |
|---------|---------|---------|
| `gymnasium` | >= 1.0 | Environment interface |
| `stable-baselines3` | >= 2.4 | PPO, A2C, DQN implementations |
| `sb3-contrib` | >= 2.4 | RecurrentPPO (LSTM policies), QRDQN |
| `optuna` | >= 4.0 | Hyperparameter optimization |
| `tensorboard` | >= 2.18 | Training visualization |
| `wandb` | >= 0.18 | Experiment tracking |

### Explainability

| Package | Version | Purpose |
|---------|---------|---------|
| `shap` | >= 0.46 | Feature importance for policy and tire models |
| `captum` | >= 0.7 | PyTorch-native attribution methods |
| `dtreeviz` | >= 2.2 | Decision tree visualization (VIPER output) |
| `matplotlib` | >= 3.9 | Plotting |
| `plotly` | >= 5.24 | Interactive visualizations |

### Deployment & Infrastructure

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | >= 1.40 | Interactive dashboard |
| `docker` | >= 27.0 | Containerization |
| `pytest` | >= 8.3 | Testing |
| `ruff` | >= 0.8 | Linting and formatting |
| `hydra-core` | >= 1.3 | Configuration management |
| `dvc` | >= 3.56 | Data version control |

---

## 7. Compute Budget

| Task | Hardware | Estimated Time | Estimated Cost |
|------|----------|---------------|----------------|
| Data ingestion & cleaning (FastF1 2022-2025) | CPU (any) | 4-8 hours (API rate limited) | Free |
| Ridge regression baseline | CPU (any) | < 5 minutes | Free |
| Bayesian hierarchical model (NumPyro NUTS) | 1x GPU (A100 or equivalent) | 2-6 hours (4 chains, 2000 warmup + 2000 samples) | $5-15 |
| XGBoost/LightGBM training + Optuna sweep | CPU (16-core) | 1-2 hours | Free / $2-5 |
| LSTM/GRU training + MC Dropout | 1x GPU (T4/A10) | 2-4 hours | $3-8 |
| RL training: PPO single-circuit (500K steps) | CPU (8-core) | 1-2 hours | Free |
| RL training: PPO multi-circuit (2-5M steps) | 1x GPU (A10/A100) | 6-12 hours | $10-25 |
| RL Optuna sweep (50 trials x 500K steps) | 1x GPU (A100) | 12-24 hours | $25-50 |
| SHAP analysis + VIPER distillation | CPU (any) | 30-60 minutes | Free |
| Multi-agent self-play (Project C, 10M+ steps) | 2-4x GPU (A100) | 24-72 hours | $50-150 |
| **Total (Projects A + B)** | | **~40-60 hours GPU** | **$50-100** |
| **Total (Projects A + B + C)** | | **~80-150 hours GPU** | **$100-250** |

Cloud options: Google Colab Pro ($12/month for A100 access), Lambda Labs ($1.10/hr A100), or RunPod ($0.74/hr A100 community).

---

## 8. Recommended Timeline

| Week | Focus | Deliverables |
|------|-------|-------------|
| **1** | Data pipeline setup | FastF1 ingestion scripts, raw Parquet files, data quality report |
| **2** | Feature engineering & EDA | Cleaned 50K-lap dataset, exploratory notebooks, feature correlation analysis |
| **3** | Tier 1 & 2 models | Ridge baseline, NumPyro hierarchical model (initial), evaluation framework |
| **4** | Tier 2 refinement | Bayesian model convergence diagnostics, posterior analysis, partial pooling validation |
| **5** | Tier 3 & 4 models | XGBoost/LightGBM with quantile regression, LSTM/GRU with MC Dropout |
| **6** | Model comparison & selection | Cross-validated benchmarks, ensemble exploration, **Project A writeup** |
| **7** | Simulator core | `gymnasium.Env` subclass, lap time dynamics, pit stop mechanics |
| **8** | Simulator extensions | Safety Car model, multi-car position tracking, regulation enforcement |
| **9** | Simulator validation | Compare simulated race outcomes vs historical results, calibrate noise parameters |
| **10** | RL training (Phase 1) | PPO on single circuit, reward shaping experiments, basic policy evaluation |
| **11** | RL training (Phase 2) | Curriculum learning, multi-circuit generalization, Optuna sweep |
| **12** | RL training (Phase 3) | Final training runs, policy convergence analysis, ablation studies |
| **13** | Explainability | SHAP analysis, VIPER decision tree, counterfactual generation |
| **14** | Validation & polish | Historical race replays, comparison vs actual strategies, **Project B writeup** |
| **15-18** | *(Optional)* Project C: Multi-agent | 2026 regs integration, opponent modeling, POMDP belief state |
| **19-22** | *(Optional)* Project C: Self-play | Self-play training, Elo evaluation, ERS strategy optimization |
| **Alt 15-16** | *(Optional)* Streamlit dashboard | Interactive strategy explorer, live race simulation, model comparison tool |

---

## 9. Key References

| Paper | Key Contribution | Data Used |
|-------|-----------------|-----------|
| Thomas et al., "Using Reinforcement Learning for Formula 1 Race Strategy," SAC '25 (arXiv:2501.04068) | PPO-based pit stop strategy agent with Gymnasium simulator; demonstrated RL can match/beat heuristic strategies | Simulated F1 environment with calibrated parameters |
| Todd et al., "A Simulation-Based Approach to Formula One Race Strategy Using Reinforcement Learning," SAC '25 (arXiv:2501.04067) | Complementary RL approach with focus on multi-car dynamics and undercut/overcut modeling | Historical race data for simulator calibration |
| Fieni et al., "Multi-Agent Reinforcement Learning for Formula One Race Strategy," Dec 2025 (arXiv:2512.21570) | Multi-agent RL formulation with shared environment; demonstrates emergent competitive strategies | Bonomi et al. tire data (Pirelli collaboration) |
| Fieni et al., "Opponent Modeling in Multi-Agent F1 Strategy," Feb 2026 (arXiv:2602.23056) | Interaction modules for opponent behavior prediction; shows improved performance over independent learning | Extended Bonomi dataset + FastF1 telemetry |
| Cappello & Hoegh, "Bayesian Tire Degradation Model for Formula 1," Nov 2025 (arXiv:2512.00640) | Bayesian state-space model for single-driver tire degradation; elegant uncertainty quantification | Single-race FastF1 data |
| Kleisarchaki, "HMM-POMDP Framework for F1 Strategy Under Partial Observability," Mar 2026 (arXiv:2603.01290) | Hidden Markov Model for inferring opponent battery state from observable sector times; POMDP policy optimization | Simulated 2026-spec environment |
| Heilmeier et al., "Race Simulation Models for Circuit Racing," Applied Sciences, 2020 | Comprehensive race simulation framework including tire degradation, fuel, and pit stop modeling | Historical F1 data |
| Heine & Thraves, "Optimal Pit Stop Strategy in Formula 1," CEJOR, 2023 | Mathematical optimization of pit stop timing; provides analytical baselines for RL comparison | Public lap time data |

---

## 10. Open-Source References

| Repository | Author | Description |
|------------|--------|-------------|
| **[Pit Stop Simulator](https://github.com/rembertdesigns/Pit-Stop-Simulator)** | rembertdesigns | Python-based F1 pit stop strategy simulator with basic tire degradation modeling. Useful reference for simulator architecture and pit stop time calculations. |
| **[Armchair Strategist](https://github.com/Casper-Guo/Armchair-Strategist)** | Casper-Guo | Data-driven F1 strategy analysis tool using FastF1 data. Includes tire stint analysis, strategy comparison visualizations, and historical race replay functionality. Good reference for FastF1 data pipeline patterns. |
| **[F1 Tire Degradation Prediction](https://github.com/schilamkur/F1-Tire-Degradation-Prediction)** | schilamkur | Machine learning approach to tire degradation prediction using historical lap time data. Implements gradient boosted models with feature engineering for tire life, fuel load, and track conditions. Useful baseline for Project A modeling approaches. |
