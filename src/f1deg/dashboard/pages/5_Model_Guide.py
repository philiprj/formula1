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
st.info(
    "All models are trained on **dry compound laps only** (SOFT, MEDIUM, HARD). "
    "Wet/intermediate tyre data is separated for future wet-specific modeling."
)

available = available_model_names()

# ---------------------------------------------------------------------------
# Tabs — one per model tier
# ---------------------------------------------------------------------------

tab_linear, tab_bayesian, tab_gbm, tab_features = st.tabs(
    [
        f"{'✅' if 'linear' in available else '⬜'} {MODEL_LABELS['linear']}",
        f"{'✅' if 'bayesian' in available else '⬜'} {MODEL_LABELS['bayesian']}",
        f"{'✅' if 'gbm' in available else '⬜'} {MODEL_LABELS['gbm']}",
        "Feature Reference",
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
| Trees | 573 (tuned) |
| Max depth | 5 (tuned) |
| Learning rate | 0.033 (tuned) |
| Quantiles | 0.025, 0.5, 0.975 |
| Typical MAE | ~1.50s (LORO) |
| Features | 36 numeric + 4 categorical |
| Dependencies | xgboost or lightgbm |
"""
        )

        st.markdown("##### Features Used")
        st.code(
            "Numeric:  tyre_life, fuel_mass_kg, track_temp,\n"
            "          air_temp, humidity, wind_speed, rainfall\n"
            "Circuit:  track_length_km, n_corners,\n"
            "          tire_stress, pit_loss_seconds\n"
            "Traffic:  position, position_change,\n"
            "          gap_ahead_seconds, gap_behind_seconds,\n"
            "          traffic_density, drs_likely\n"
            "Stint:    race_progress, stint_fraction\n"
            "Weekend:  circuit_baseline_pace,\n"
            "          driver_pace_vs_field, fp_deg_rate_*,\n"
            "          quali_position, weekend_track_temp\n"
            "Rolling:  lap_time_delta, deg_rate_estimate,\n"
            "          lap_time_rolling_std, tyre_life_cubed\n"
            "SC/Flags: laps_since_sc_end, had_sc_this_stint\n"
            "Interact: compound_x_track_temp,\n"
            "          tyre_life_x_compound\n"
            "Categorical: compound, circuit_id,\n"
            "             driver_id, constructor_id",
            language=None,
        )
        st.caption("See the **Feature Reference** tab for details on each feature.")

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
# Feature Reference
# ---------------------------------------------------------------------------

with tab_features:
    st.header("Feature Reference")
    st.markdown(
        "A comprehensive guide to every feature used across the model tiers. "
        "Features are grouped by category. The **Model** column shows which "
        "tiers use each feature: **L** = Linear, **B** = Bayesian, **G** = GBM."
    )

    # --- Core tyre / fuel ---
    st.subheader("Core: Tyre & Fuel")
    st.markdown(
        """
| Feature | Description | Units | Model |
| --- | --- | --- | --- |
| `tyre_life` | Laps since the current set of tyres was fitted | laps | L B G |
| `tyre_life_sq` | `tyre_life²` — captures accelerating degradation (quadratic term) | laps² | L B |
| `tyre_life_cubed` | `tyre_life³` — captures the "tyre cliff" (computed at predict time in GBM) | laps³ | G |
| `fuel_mass_kg` | Estimated fuel remaining, decreasing ~1.6 kg/lap from ~110 kg start | kg | L B G |
| `compound` | Tyre compound: SOFT, MEDIUM, or HARD (label-encoded for GBM, one-hot for Linear) | categorical | L B G |
"""
    )

    # --- Weather ---
    st.subheader("Weather Conditions")
    st.markdown(
        """
| Feature | Description | Units | Model |
| --- | --- | --- | --- |
| `track_temp` | Track surface temperature from FIA sensors | °C | L G |
| `air_temp` | Ambient air temperature | °C | L G |
| `humidity` | Relative humidity | % | G |
| `wind_speed` | Wind speed at track level | m/s | G |
| `rainfall` | Precipitation intensity (0 for most dry laps, but light rain is possible) | mm/h | L G |
"""
    )

    # --- Circuit characteristics ---
    st.subheader("Circuit Characteristics")
    st.markdown(
        """
These numeric features encode the physical properties of each circuit, allowing
the model to learn track-specific behaviour without relying solely on the
`circuit_id` categorical. Added to replace `track_rubber_index`.

| Feature | Description | Units | Model |
| --- | --- | --- | --- |
| `track_length_km` | Total lap distance | km | G |
| `n_corners` | Number of corners per lap (from circuit data) | count | G |
| `tire_stress` | Composite metric: `n_corners / track_length_km` — higher values mean more cornering load per km | corners/km | G |
| `pit_loss_seconds` | Time lost entering and exiting the pit lane vs staying on track | seconds | G |
| `circuit_id` | Categorical circuit identifier (label-encoded for GBM, one-hot for Linear) | categorical | L B G |

**Feature importance insight:** `track_length_km` and `n_corners` are the
2nd and 3rd most important features by gain (16.5% and 9.1% respectively),
together contributing more predictive power than the `circuit_id` categorical alone.
"""
    )

    # --- Position & traffic ---
    st.subheader("Position & Traffic")
    st.markdown(
        """
| Feature | Description | Units | Model |
| --- | --- | --- | --- |
| `position` | Race position at the start of this lap | int | G |
| `position_change` | Positions gained/lost on the previous lap (+ve = gained) | int | G |
| `gap_ahead_seconds` | Time gap to the car ahead | seconds | G |
| `gap_behind_seconds` | Time gap to the car behind | seconds | G |
| `traffic_density` | Number of cars within 1.5 seconds | count | G |
| `drs_likely` | Whether DRS is likely available (gap < 1s to car ahead) | 0/1 | G |
"""
    )

    # --- Stint context ---
    st.subheader("Stint & Race Context")
    st.markdown(
        """
| Feature | Description | Units | Model |
| --- | --- | --- | --- |
| `race_progress` | Fraction of total race distance completed (0 to 1) | ratio | G |
| `stint_fraction` | Fraction of expected stint length completed | ratio | G |
| `laps_since_sc_end` | Laps elapsed since the most recent safety car period ended | laps | G |
| `laps_since_red_flag` | Laps elapsed since the most recent red flag restart | laps | G |
| `had_sc_this_stint` | Whether a safety car occurred during the current stint | 0/1 | G |
"""
    )

    # --- Weekend calibration ---
    st.subheader("Weekend Calibration")
    st.markdown(
        """
These features are derived from practice/qualifying sessions earlier in the
weekend, giving the model track-specific and driver-specific context before the
race starts.

| Feature | Description | Units | Model |
| --- | --- | --- | --- |
| `circuit_baseline_pace` | Median clean lap time from FP sessions — sets the base pace for this circuit/weekend | seconds | G |
| `driver_pace_vs_field` | Driver's FP pace relative to the field median (negative = faster) | seconds | G |
| `fp_deg_rate_soft` | Degradation rate for SOFT compound estimated from FP long runs | s/lap | G |
| `fp_deg_rate_medium` | Degradation rate for MEDIUM compound estimated from FP long runs | s/lap | G |
| `fp_deg_rate_hard` | Degradation rate for HARD compound estimated from FP long runs | s/lap | G |
| `weekend_track_temp` | Average track temperature across FP sessions | °C | G |
| `quali_position` | Qualifying position (grid slot) — proxy for car/driver performance | int | G |
| `expected_fp3_race_delta` | Expected pace difference between FP3 and race conditions | seconds | G |

**Feature importance insight:** `circuit_baseline_pace` is the single most
important feature at 27.2% of gain — it gives the model the absolute lap-time
scale for each circuit/weekend.
"""
    )

    # --- Interaction features ---
    st.subheader("Interaction Features")
    st.markdown(
        """
Pre-computed interactions that help models capture known physical relationships.

| Feature | Description | Units | Model |
| --- | --- | --- | --- |
| `compound_x_track_temp` | compound_encoded * track_temp — captures compound-specific temperature sensitivity | - | G |
| `tyre_life_x_compound` | tyre_life * compound_encoded — captures compound-specific degradation rate | - | G |
"""
    )

    # --- Rolling / lag features ---
    st.subheader("Rolling & Lag Features (GBM only)")
    st.markdown(
        """
Computed at prediction time within each stint (5-lap rolling window). These
give the model real-time information about how lap times are evolving.

| Feature | Description | Units | Model |
| --- | --- | --- | --- |
| `lap_time_delta` | Current lap time minus the rolling median (5-lap window) | seconds | G |
| `lap_time_rolling_std` | Rolling standard deviation of lap times (5-lap window) | seconds | G |
| `deg_rate_estimate` | Lap-over-lap time increase (first difference) | seconds | G |
"""
    )

    # --- Categorical identifiers ---
    st.subheader("Categorical Identifiers")
    st.markdown(
        """
| Feature | Description | Encoding | Model |
| --- | --- | --- | --- |
| `compound` | Tyre compound (SOFT / MEDIUM / HARD) | One-hot (L, B) or Label (G) | L B G |
| `circuit_id` | Circuit identifier (e.g. `monza`, `silverstone`) | One-hot (L, B) or Label (G) | L B G |
| `driver_id` | Driver identifier (e.g. `max_verstappen`) | Label (G) | B G |
| `constructor_id` | Constructor/team identifier (e.g. `red_bull`) | Label (G) | G |

**Feature importance insight:** `compound` and `driver_id` have very low
importance in the GBM (< 0.5% each) because `fp_deg_rate_*` and
`driver_pace_vs_field` capture the same information more directly.
"""
    )

    # --- Model usage matrix ---
    st.divider()
    st.subheader("Feature Usage by Model")

    feature_matrix = {
        "Feature": [
            "tyre_life",
            "tyre_life_sq",
            "tyre_life_cubed",
            "fuel_mass_kg",
            "track_temp",
            "air_temp",
            "humidity",
            "wind_speed",
            "rainfall",
            "track_length_km",
            "n_corners",
            "tire_stress",
            "pit_loss_seconds",
            "position",
            "gap_ahead_seconds",
            "traffic_density",
            "drs_likely",
            "race_progress",
            "stint_fraction",
            "circuit_baseline_pace",
            "fp_deg_rate_*",
            "quali_position",
            "compound_x_track_temp",
            "lap_time_delta",
            "deg_rate_estimate",
            "laps_since_sc_end",
            "had_sc_this_stint",
            "compound",
            "circuit_id",
            "driver_id",
            "constructor_id",
        ],
        "Linear": [
            "Y",
            "Y",
            "",
            "Y",
            "Y",
            "Y",
            "",
            "",
            "Y",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "Y",
            "Y",
            "",
            "",
        ],
        "Bayesian": [
            "Y",
            "Y",
            "",
            "Y",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "Y",
            "Y",
            "Y",
            "",
        ],
        "GBM": [
            "Y",
            "",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
        ],
    }

    import pandas as pd

    matrix_df = pd.DataFrame(feature_matrix)
    # Replace Y/empty with checkmarks
    for col in ["Linear", "Bayesian", "GBM"]:
        matrix_df[col] = matrix_df[col].apply(lambda x: "✅" if x == "Y" else "")

    st.dataframe(matrix_df, use_container_width=True, hide_index=True, height=800)


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

st.divider()
st.header("Model Comparison")

st.markdown(
    """
| | Linear (Ridge) | Bayesian Hierarchical | Gradient Boosted Trees |
| --- | --- | --- | --- |
| **Typical MAE** | ~1.54s | ~1.33s | ~1.50s (LORO) |
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
