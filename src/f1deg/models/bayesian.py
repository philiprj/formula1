"""Tier 2: Bayesian hierarchical state-space model for tire degradation.

Extends Cappello & Hoegh (2024) from single-driver/single-race to
multi-driver/multi-race with partial pooling across circuits and compounds.

Implemented in NumPyro (JAX backend).
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from f1deg.models.base import DegradationModel

# Lazy imports for optional dependency
_numpyro = None
_jax = None


def _import_numpyro():
    global _numpyro, _jax
    if _numpyro is None:
        import jax
        import numpyro

        _jax = jax
        _numpyro = numpyro
    return _numpyro, _jax


class BayesianDegradationModel(DegradationModel):
    """Bayesian hierarchical state-space model for tire degradation.

    Model structure:
        y[t] = pace[t] + fuel_effect * fuel[t] + driver_offset[d] + circuit_offset[c] + eps
        pace[t] = pace[t-1] + deg_rate[compound, circuit] + eta
        deg_rate[compound, circuit] ~ Normal(mu_deg[compound], sigma_deg_circuit)

    Uses NumPyro for inference (NUTS MCMC or SVI for fast iteration).
    """

    def __init__(self):
        self.samples: dict | None = None
        self.svi_params: dict | None = None
        self.config: dict = {}
        self.encoders: dict = {}  # Maps categorical values to integer indices

    def fit(self, train_df: pd.DataFrame, config: dict) -> None:
        numpyro, jax = _import_numpyro()
        from numpyro.infer import MCMC, NUTS

        self.config = config
        model_config = config.get("model", {})
        mcmc_config = model_config.get("mcmc", {})

        # Enable parallel chains on CPU
        numpyro.set_host_device_count(mcmc_config.get("num_chains", 4))

        # Encode categoricals to integer indices
        data = self._encode_data(train_df)

        # Run MCMC
        kernel = NUTS(
            self._model,
            target_accept_prob=mcmc_config.get("target_accept_prob", 0.85),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=mcmc_config.get("num_warmup", 1000),
            num_samples=mcmc_config.get("num_samples", 2000),
            num_chains=mcmc_config.get("num_chains", 4),
        )

        rng_key = jax.random.PRNGKey(0)
        mcmc.run(rng_key, **data)
        self.samples = mcmc.get_samples()

    def fit_svi(self, train_df: pd.DataFrame, config: dict) -> None:
        """Fast variational inference (for CV and development iteration).

        After fitting, draws samples from the variational posterior and stores
        them in self.samples so that predict() / predict_interval() work
        identically to MCMC.
        """
        numpyro, jax = _import_numpyro()
        from numpyro.infer import SVI, Predictive, Trace_ELBO, autoguide

        self.config = config
        model_config = config.get("model", {})
        svi_config = model_config.get("svi", {})

        data = self._encode_data(train_df)

        guide = autoguide.AutoNormal(self._model)
        optimizer = numpyro.optim.Adam(svi_config.get("learning_rate", 0.005))
        svi = SVI(self._model, guide, optimizer, loss=Trace_ELBO())

        rng_key = jax.random.PRNGKey(0)
        svi_result = svi.run(
            rng_key,
            svi_config.get("num_steps", 10000),
            **data,
        )
        self.svi_params = svi_result.params

        # Draw samples from variational posterior so predict() works
        predictive = Predictive(
            self._model,
            guide=guide,
            params=self.svi_params,
            num_samples=500,
            return_sites=[
                "global_pace",
                "circuit_offset_sd",
                "circuit_offset_raw",
                "circuit_offset",
                "base_pace",
                "deg_rate",
                "fuel_effect",
                "driver_offset_sd",
                "driver_offset_raw",
                "driver_offset",
                "track_temp_effect",
                "race_progress_effect",
                "drs_effect",
                "sigma_obs",
            ],
        )
        rng_key = jax.random.PRNGKey(1)
        self.samples = {k: np.asarray(v) for k, v in predictive(rng_key, **data).items()}

    def _get_index_arrays(self, data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert JAX index arrays to numpy for vectorized prediction."""
        return (
            np.asarray(data["compound_idx"]),
            np.asarray(data["circuit_idx"]),
            np.asarray(data["driver_idx"]),
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.samples is None and self.svi_params is None:
            raise RuntimeError("Model has not been fitted. Call fit() or fit_svi() first.")

        data = self._encode_data(df)
        compound_idx, circuit_idx, driver_idx = self._get_index_arrays(data)
        tyre_life = np.asarray(data["tyre_life"])
        fuel_mass = np.asarray(data["fuel_mass"])

        if self.samples is not None:
            # Use posterior mean — fully vectorized
            global_pace = float(np.mean(self.samples["global_pace"]))
            fuel_effect = float(np.mean(self.samples["fuel_effect"]))
            deg_rates = np.mean(self.samples["deg_rate"], axis=0)
            base_pace = np.mean(self.samples["base_pace"], axis=0)
            driver_offsets = np.mean(self.samples["driver_offset"], axis=0)
            circuit_offsets = np.mean(self.samples["circuit_offset"], axis=0)
        else:
            raise NotImplementedError("SVI prediction not yet implemented")

        predictions = (
            global_pace
            + circuit_offsets[circuit_idx]
            + base_pace[compound_idx]
            + deg_rates[compound_idx] * tyre_life
            + fuel_effect * fuel_mass
            + driver_offsets[driver_idx]
        )

        # Add covariate effects using posterior means
        if "track_temp" in data and "track_temp_effect" in self.samples:
            track_temp_eff = float(np.mean(self.samples["track_temp_effect"]))
            predictions = predictions + track_temp_eff * np.asarray(data["track_temp"])
        if "race_progress" in data and "race_progress_effect" in self.samples:
            rp_eff = float(np.mean(self.samples["race_progress_effect"]))
            predictions = predictions + rp_eff * np.asarray(data["race_progress"])
        if "drs_likely" in data and "drs_effect" in self.samples:
            drs_eff = float(np.mean(self.samples["drs_effect"]))
            predictions = predictions + drs_eff * np.asarray(data["drs_likely"])

        return predictions

    def predict_interval(
        self, df: pd.DataFrame, alpha: float = 0.05, max_samples: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.samples is None:
            pred = self.predict(df)
            margin = 1.5
            return pred - margin, pred + margin

        data = self._encode_data(df)
        compound_idx, circuit_idx, driver_idx = self._get_index_arrays(data)
        tyre_life = np.asarray(data["tyre_life"])
        fuel_mass = np.asarray(data["fuel_mass"])

        # Subsample MCMC draws to keep memory reasonable
        n_total = len(self.samples["fuel_effect"])
        if n_total > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_total, size=max_samples, replace=False)
        else:
            idx = np.arange(n_total)

        # Vectorized: (n_samples, n_laps) via broadcasting
        gp = np.asarray(self.samples["global_pace"])[idx, None]  # (S, 1)
        fuel_eff = np.asarray(self.samples["fuel_effect"])[idx, None]  # (S, 1)
        deg = np.asarray(self.samples["deg_rate"])[idx][:, compound_idx]  # (S, N)
        base = np.asarray(self.samples["base_pace"])[idx][:, compound_idx]  # (S, N)
        drv = np.asarray(self.samples["driver_offset"])[idx][:, driver_idx]  # (S, N)
        cir = np.asarray(self.samples["circuit_offset"])[idx][:, circuit_idx]  # (S, N)

        all_predictions = (
            gp + cir + base + deg * tyre_life[None, :] + fuel_eff * fuel_mass[None, :] + drv
        )

        # Add covariate effects with full posterior uncertainty
        if "track_temp" in data and "track_temp_effect" in self.samples:
            tt_eff = np.asarray(self.samples["track_temp_effect"])[idx, None]  # (S, 1)
            tt_vals = np.asarray(data["track_temp"])[None, :]  # (1, N)
            all_predictions = all_predictions + tt_eff * tt_vals
        if "race_progress" in data and "race_progress_effect" in self.samples:
            rp_eff = np.asarray(self.samples["race_progress_effect"])[idx, None]
            rp_vals = np.asarray(data["race_progress"])[None, :]
            all_predictions = all_predictions + rp_eff * rp_vals
        if "drs_likely" in data and "drs_effect" in self.samples:
            drs_eff = np.asarray(self.samples["drs_effect"])[idx, None]
            drs_vals = np.asarray(data["drs_likely"])[None, :]
            all_predictions = all_predictions + drs_eff * drs_vals

        # Add observation noise for proper *prediction* intervals (not just parameter uncertainty)
        if "sigma_obs" in self.samples:
            sigma = np.asarray(self.samples["sigma_obs"])[idx, None]  # (S, 1)
            rng = np.random.default_rng(42)
            noise = rng.normal(0, sigma, size=all_predictions.shape)
            all_predictions = all_predictions + noise

        lower = np.percentile(all_predictions, 100 * alpha / 2, axis=0)
        upper = np.percentile(all_predictions, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(
                {
                    "samples": {k: np.array(v) for k, v in self.samples.items()}
                    if self.samples
                    else None,
                    "encoders": self.encoders,
                    "config": self.config,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "BayesianDegradationModel":
        instance = cls()
        with open(path / "model.pkl", "rb") as f:
            data = pickle.load(f)
        instance.samples = data["samples"]
        instance.encoders = data["encoders"]
        instance.config = data["config"]
        return instance

    def _model(
        self,
        tyre_life,
        fuel_mass,
        compound_idx,
        circuit_idx,
        driver_idx,
        n_compounds,
        n_circuits,
        n_drivers,
        track_temp=None,
        race_progress=None,
        drs_likely=None,
        y=None,
    ):
        """NumPyro model specification.

        Parameterization:
            y[t] = circuit_pace[c] + compound_offset[k] + deg_rate[k] * tyre_life
                   + fuel_effect * fuel_mass + driver_offset[d]
                   + track_temp_effect * track_temp
                   + race_progress_effect * race_progress
                   + drs_effect * drs_likely
                   + eps

        circuit_pace absorbs the absolute lap time (~70-111s) per circuit.
        compound_offset is relative to circuit pace (SOFT faster, HARD slower).
        deg_rate is degradation in s/lap per compound.
        fuel_effect is constrained near physics (~0.035 s/kg).
        driver_offset is relative skill.
        track_temp_effect captures grip changes with temperature.
        race_progress_effect captures rubber buildup improving grip over the race.
        drs_effect captures the DRS time advantage.
        """
        numpyro, _jax = _import_numpyro()
        import numpyro.distributions as dist

        priors = self.config.get("model", {}).get("priors", {})

        # Circuit pace — this is the BIG number (~70-111s depending on circuit)
        # Hierarchical: global mean + per-circuit deviation
        global_pace = numpyro.sample(
            "global_pace",
            dist.Normal(
                priors.get("global_pace_mean", 90.0),
                priors.get("global_pace_sd", 10.0),
            ),
        )
        circuit_offset_sd = numpyro.sample(
            "circuit_offset_sd",
            dist.HalfNormal(priors.get("circuit_offset_scale", 15.0)),
        )
        circuit_offset_raw = numpyro.sample(
            "circuit_offset_raw",
            dist.Normal(0.0, 1.0).expand([n_circuits]),
        )
        # Non-centered parameterization for better sampling
        circuit_offset = numpyro.deterministic(
            "circuit_offset", circuit_offset_raw * circuit_offset_sd
        )

        # Compound offset — relative to circuit pace (SOFT < 0, HARD > 0)
        # Use base_pace name for backward compatibility with predict()
        base_pace = numpyro.sample(
            "base_pace",
            dist.Normal(
                priors.get("compound_offset_mean", 0.0),
                priors.get("compound_offset_sd", 3.0),
            ).expand([n_compounds]),
        )

        # Degradation rate — s/lap, positive means getting slower
        # Tighter prior informed by physics
        deg_rate = numpyro.sample(
            "deg_rate",
            dist.LogNormal(
                priors.get("deg_rate_log_mean", -3.0),  # exp(-3) ≈ 0.05 s/lap
                priors.get("deg_rate_log_sd", 0.5),
            ).expand([n_compounds]),
        )

        # Fuel effect — strongly constrained by physics (~0.035 s/kg)
        fuel_effect = numpyro.sample(
            "fuel_effect",
            dist.Normal(
                priors.get("fuel_effect_mean", 0.035),
                priors.get("fuel_effect_sd", 0.005),
            ),
        )

        # Driver offset — hierarchical with learned scale
        driver_offset_sd = numpyro.sample(
            "driver_offset_sd",
            dist.HalfNormal(priors.get("driver_offset_scale", 1.5)),
        )
        driver_offset_raw = numpyro.sample(
            "driver_offset_raw",
            dist.Normal(0.0, 1.0).expand([n_drivers]),
        )
        driver_offset = numpyro.deterministic("driver_offset", driver_offset_raw * driver_offset_sd)

        # --- Covariate effects ---

        # Track temperature effect: hotter track = more grip (faster laps, negative coeff)
        # Centered at -0.02 s/°C — small but consistent effect
        track_temp_effect = numpyro.sample(
            "track_temp_effect",
            dist.Normal(
                priors.get("track_temp_effect_mean", -0.02),
                priors.get("track_temp_effect_sd", 0.03),
            ),
        )

        # Race progress effect: rubber buildup makes track faster (negative coeff)
        # Typical effect: ~0.3-0.8s faster from start to end of race
        race_progress_effect = numpyro.sample(
            "race_progress_effect",
            dist.Normal(
                priors.get("race_progress_effect_mean", -0.5),
                priors.get("race_progress_effect_sd", 0.5),
            ),
        )

        # DRS effect: having DRS makes a lap faster (negative coeff)
        # Typical DRS advantage: 0.3-0.6s per lap
        drs_effect = numpyro.sample(
            "drs_effect",
            dist.Normal(
                priors.get("drs_effect_mean", -0.4),
                priors.get("drs_effect_sd", 0.3),
            ),
        )

        # Observation noise
        sigma_obs = numpyro.sample(
            "sigma_obs",
            dist.HalfNormal(priors.get("sigma_obs_scale", 2.0)),
        )

        # Observation model
        mu = (
            global_pace
            + circuit_offset[circuit_idx]
            + base_pace[compound_idx]
            + deg_rate[compound_idx] * tyre_life
            + fuel_effect * fuel_mass
            + driver_offset[driver_idx]
        )

        # Add covariate effects (covariates are z-scored in _encode_data)
        if track_temp is not None:
            mu = mu + track_temp_effect * track_temp
        if race_progress is not None:
            mu = mu + race_progress_effect * race_progress
        if drs_likely is not None:
            mu = mu + drs_effect * drs_likely

        obs_df = priors.get("obs_df", 5.0)
        numpyro.sample(
            "obs",
            dist.StudentT(obs_df, mu, sigma_obs),
            obs=y,
        )

    def _encode_data(self, df: pd.DataFrame) -> dict:
        """Encode DataFrame into arrays suitable for the NumPyro model."""
        import jax.numpy as jnp

        # Build or reuse encoders
        for col, _out_col in [
            ("compound", "compound_idx"),
            ("circuit_id", "circuit_idx"),
            ("driver_id", "driver_idx"),
        ]:
            if col not in self.encoders and col in df.columns:
                unique_vals = sorted(df[col].unique())
                self.encoders[col] = {v: i for i, v in enumerate(unique_vals)}

        def encode_col(series, encoder):
            return np.array([encoder.get(v, 0) for v in series])

        compound_idx = encode_col(
            df.get("compound", pd.Series(["MEDIUM"] * len(df))),
            self.encoders.get("compound", {"MEDIUM": 0}),
        )
        circuit_idx = encode_col(
            df.get("circuit_id", pd.Series(["unknown"] * len(df))),
            self.encoders.get("circuit_id", {"unknown": 0}),
        )
        driver_idx = encode_col(
            df.get("driver_id", pd.Series(["UNK"] * len(df))),
            self.encoders.get("driver_id", {"UNK": 0}),
        )

        data = {
            "tyre_life": jnp.array(
                df["tyre_life"].values if "tyre_life" in df.columns else np.ones(len(df)),
                dtype=jnp.float32,
            ),
            "fuel_mass": jnp.array(
                df["fuel_mass_kg"].values
                if "fuel_mass_kg" in df.columns
                else np.full(len(df), 80.0),
                dtype=jnp.float32,
            ),
            "compound_idx": jnp.array(compound_idx, dtype=jnp.int32),
            "circuit_idx": jnp.array(circuit_idx, dtype=jnp.int32),
            "driver_idx": jnp.array(driver_idx, dtype=jnp.int32),
            "n_compounds": len(self.encoders.get("compound", {"MEDIUM": 0})),
            "n_circuits": len(self.encoders.get("circuit_id", {"unknown": 0})),
            "n_drivers": len(self.encoders.get("driver_id", {"UNK": 0})),
        }

        # Covariate features — z-score normalize to help MCMC sampling
        if "track_temp" in df.columns:
            vals = df["track_temp"].fillna(30.0).values.astype(float)
            # Store stats for prediction-time normalization
            if "track_temp_stats" not in self.encoders:
                self.encoders["track_temp_stats"] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)) or 1.0,
                }
            stats = self.encoders["track_temp_stats"]
            data["track_temp"] = jnp.array((vals - stats["mean"]) / stats["std"], dtype=jnp.float32)

        if "race_progress" in df.columns:
            vals = df["race_progress"].fillna(0.5).values.astype(float)
            data["race_progress"] = jnp.array(vals, dtype=jnp.float32)

        if "drs_likely" in df.columns:
            vals = df["drs_likely"].fillna(0.0).values.astype(float)
            data["drs_likely"] = jnp.array(vals, dtype=jnp.float32)

        if "lap_time_seconds" in df.columns:
            data["y"] = jnp.array(df["lap_time_seconds"].values, dtype=jnp.float32)
        else:
            data["y"] = None

        return data
