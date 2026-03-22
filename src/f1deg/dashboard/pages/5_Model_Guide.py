"""Page 5: Model Guide — explains each degradation model tier."""

from __future__ import annotations

import streamlit as st

from f1deg.dashboard.state import MODEL_LABELS, available_model_names

st.set_page_config(page_title="Model Guide", layout="wide")
st.title("Model Guide")
st.markdown(
    "An overview of the degradation models available in the dashboard, "
    "how they work, and when to use each one."
)

available = available_model_names()

# ---------------------------------------------------------------------------
# Tabs — one per model tier
# ---------------------------------------------------------------------------

tab_linear, tab_bayesian, tab_gbm = st.tabs(
    [
        f"{'✅' if 'linear' in available else '⬜'} {MODEL_LABELS['linear']}",
        f"{'✅' if 'bayesian' in available else '⬜'} {MODEL_LABELS['bayesian']}",
        f"{'✅' if 'gbm' in available else '⬜'} {MODEL_LABELS['gbm']}",
    ]
)

# ---------------------------------------------------------------------------
# Tier 1: Linear (Ridge)
# ---------------------------------------------------------------------------

with tab_linear:
    st.header("Tier 1: Linear (Ridge Regression)")

    col_overview, col_details = st.columns([3, 2])

    with col_overview:
        st.markdown(
            """
**What it does**

A Ridge regression baseline that models lap time as a linear combination of
tyre life, fuel load, weather conditions, and categorical identifiers for
compound and circuit.

**How it works**

The model fits the equation:

```
lap_time = w₁·tyre_life + w₂·tyre_life² + w₃·fuel_mass
         + w₄·track_temp + w₅·air_temp + w₆·rainfall
         + compound_offsets + circuit_offsets + intercept
```

Categorical features (compound, circuit) are one-hot encoded before fitting.
Ridge regularisation (L2 penalty, default a=1.0) prevents overfitting by
shrinking coefficients.

**Prediction intervals** are computed using the residual standard error,
assuming normally distributed errors. This gives constant-width intervals
that don't adapt to local conditions.
"""
        )

    with col_details:
        st.markdown("##### Key Properties")
        st.markdown(
            """
| Property | Value |
| --- | --- |
| Algorithm | Ridge regression (sklearn) |
| Regularisation | L2 (a=1.0) |
| Encoding | One-hot for categoricals |
| Interval method | Residual std (normal) |
| Typical MAE | ~1.54s |
| Dependencies | scikit-learn |
"""
        )

        st.markdown("##### Features Used")
        st.code(
            "Numeric:  tyre_life, tyre_life_sq, fuel_mass_kg,\n"
            "          track_temp, air_temp, rainfall\n"
            "Categorical: compound, circuit_id",
            language=None,
        )

    st.divider()

    st.markdown("##### When to use")
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown(
            """
**Strengths**
- Fast to train and predict (milliseconds)
- Fully interpretable — coefficients map directly to effects
- No optional dependencies required
- Good starting point / sanity-check baseline
"""
        )
    with col_con:
        st.markdown(
            """
**Limitations**
- Cannot capture non-linear compound/circuit interactions
- Fixed-width prediction intervals (not condition-aware)
- No driver or constructor effects
- Highest MAE of the three tiers
"""
        )


# ---------------------------------------------------------------------------
# Tier 2: Bayesian Hierarchical
# ---------------------------------------------------------------------------

with tab_bayesian:
    st.header("Tier 2: Bayesian Hierarchical State-Space")

    col_overview, col_details = st.columns([3, 2])

    with col_overview:
        st.markdown(
            """
**What it does**

A Bayesian hierarchical model that decomposes lap time into physically
meaningful components — circuit pace, compound offset, degradation rate,
fuel effect, and driver skill — with partial pooling across groups.

**How it works**

The generative model is:

```
y[t] = global_pace + circuit_offset[c] + compound_offset[k]
     + deg_rate[k] · tyre_life + fuel_effect · fuel_mass
     + driver_offset[d] + ε
```

Each component has an informative prior based on F1 physics:
- **Fuel effect** is tightly constrained near 0.035 s/kg (a physical constant)
- **Degradation rate** uses a LogNormal prior to enforce positivity
- **Circuit/driver offsets** use non-centered parameterisations for efficient
  sampling

The model uses **partial pooling** — individual circuit/driver estimates
borrow strength from the group mean, which helps when data is sparse
(e.g. a rookie driver with few races).

**Inference** runs via NUTS MCMC (4 chains, 2000 samples after warmup) or
SVI for fast iteration during cross-validation.

**Prediction intervals** come directly from posterior samples — they are
naturally asymmetric and condition-aware, widening where the model is
genuinely uncertain.
"""
        )

    with col_details:
        st.markdown("##### Key Properties")
        st.markdown(
            """
| Property | Value |
| --- | --- |
| Framework | NumPyro (JAX backend) |
| Inference | MCMC (NUTS) or SVI |
| Chains / Samples | 4 / 2000 |
| Observation noise | Student-t (df=5) |
| Partial pooling | Circuits, compounds, drivers |
| Typical MAE | ~1.33s |
| Dependencies | numpyro, jax |
"""
        )

        st.markdown("##### Prior Choices")
        st.code(
            "global_pace     ~ N(90, 10)\n"
            "circuit_offset  ~ N(0, s_circuit)  [non-centered]\n"
            "compound_offset ~ N(0, 3)\n"
            "deg_rate        ~ LogNormal(-3, 0.5)  [~0.05 s/lap]\n"
            "fuel_effect     ~ N(0.035, 0.005)\n"
            "driver_offset   ~ N(0, s_driver)  [non-centered]\n"
            "s_obs           ~ HalfNormal(2)",
            language=None,
        )

    st.divider()

    st.markdown("##### When to use")
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown(
            """
**Strengths**
- Physics-informed priors encode domain knowledge
- Partial pooling handles sparse data gracefully
- Posterior intervals reflect genuine uncertainty
- Student-t likelihood is robust to outliers
- Interpretable decomposition of lap time components
"""
        )
    with col_con:
        st.markdown(
            """
**Limitations**
- Slower to train (minutes for MCMC, seconds for SVI)
- Requires JAX + NumPyro (optional dependencies)
- Linear degradation rate (no non-linear tyre cliff)
- Higher MAE than the GBM tier
"""
        )


# ---------------------------------------------------------------------------
# Tier 3: Gradient Boosted Trees
# ---------------------------------------------------------------------------

with tab_gbm:
    st.header("Tier 3: Gradient Boosted Trees")

    col_overview, col_details = st.columns([3, 2])

    with col_overview:
        st.markdown(
            """
**What it does**

An ensemble of decision trees trained via gradient boosting. This tier
prioritises raw predictive accuracy and can capture complex, non-linear
interactions between features that the linear and Bayesian models miss.

**How it works**

Three separate models are trained using **quantile regression**:
1. **Median model** (q=0.5) — point predictions
2. **Lower model** (q=0.025) — lower bound of the 95% prediction interval
3. **Upper model** (q=0.975) — upper bound

Each model is an ensemble of 500 shallow decision trees (max depth 6),
trained sequentially. Each new tree corrects the errors of the previous
ensemble. Regularisation is applied via:
- **Subsampling** (80% of rows and columns per tree)
- **L1/L2 penalties** on leaf weights
- **Minimum child weight** to prevent overfitting small groups

Categorical features are label-encoded (integer mapped) rather than
one-hot encoded, which is more efficient for high-cardinality features
like `driver_id`.

**Prediction intervals** come from the quantile models — unlike the
Bayesian approach, these don't require sampling and are fast to compute.
They naturally adapt to local conditions (wider where the training data
is sparse or noisy).
"""
        )

    with col_details:
        st.markdown("##### Key Properties")
        st.markdown(
            """
| Property | Value |
| --- | --- |
| Backend | XGBoost (default) or LightGBM |
| Trees | 500 |
| Max depth | 6 |
| Learning rate | 0.05 |
| Quantiles | 0.025, 0.5, 0.975 |
| Typical MAE | ~0.84s |
| Dependencies | xgboost or lightgbm |
"""
        )

        st.markdown("##### Features Used")
        st.code(
            "Numeric:  tyre_life, tyre_life_sq, fuel_mass_kg,\n"
            "          track_temp, air_temp, humidity,\n"
            "          wind_speed, rainfall\n"
            "Traffic:  position, position_change,\n"
            "          gap_ahead_seconds, gap_behind_seconds,\n"
            "          traffic_density\n"
            "Stint:    race_progress, stint_fraction\n"
            "Interact: compound_x_track_temp,\n"
            "          tyre_life_x_track_temp\n"
            "Categorical: compound, circuit_id,\n"
            "             driver_id, constructor_id",
            language=None,
        )

    st.divider()

    st.markdown("##### When to use")
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown(
            """
**Strengths**
- Best point accuracy (lowest MAE)
- Captures non-linear interactions automatically
- Fast training and inference
- Condition-adaptive prediction intervals
- Handles high-cardinality categoricals efficiently
"""
        )
    with col_con:
        st.markdown(
            """
**Limitations**
- Less interpretable than Linear or Bayesian tiers
- No built-in physical constraints (could extrapolate oddly)
- Quantile intervals can occasionally cross
- Requires XGBoost or LightGBM dependency
"""
        )


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

st.divider()
st.header("Model Comparison")

st.markdown(
    """
| | Linear (Ridge) | Bayesian Hierarchical | Gradient Boosted Trees |
| --- | --- | --- | --- |
| **Typical MAE** | ~1.54s | ~1.33s | ~0.84s |
| **Training time** | Seconds | Minutes (MCMC) | Seconds |
| **Prediction speed** | Fast | Moderate | Fast |
| **Interpretability** | High | High | Low |
| **Uncertainty quality** | Fixed-width (normal) | Posterior samples | Quantile regression |
| **Physics-informed** | No | Yes (priors) | No |
| **Sparse data handling** | Poor | Good (pooling) | Moderate |
| **Required dependencies** | scikit-learn | numpyro, jax | xgboost / lightgbm |

**Recommendation**: Use the **GBM** tier when accuracy matters most (e.g. strategy
simulation). Use **Bayesian** when you need principled uncertainty estimates or
interpretable components. Use **Linear** as a fast baseline or when you want
fully transparent coefficients.
"""
)
