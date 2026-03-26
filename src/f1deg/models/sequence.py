"""Tier 4: LSTM/GRU sequence model for tire degradation.

Operates over stint-level sequences with MC dropout for uncertainty estimation.
Includes a degradation-rate auxiliary head and stint-lap weighted loss to
prioritise late-stint cliff detection.
"""

import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from f1deg.models.base import DegradationModel

logger = logging.getLogger(__name__)

# Lazy import for optional torch dependency
_torch = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


class StintSequenceNet:
    """PyTorch LSTM/GRU network for stint-level lap time prediction.

    Supports an optional auxiliary head that predicts lap-to-lap delta
    (degradation rate), forcing the network to learn the rate of change.
    """

    def __init__(self, config: dict, vocab_sizes: dict[str, int] | None = None):
        _import_torch()
        import torch.nn as nn

        self.config = config
        arch = config.get("architecture", "lstm")
        hidden_size = config.get("hidden_size", 64)
        num_layers = config.get("num_layers", 1)
        dropout = config.get("dropout", 0.3)

        embedding_dims = config.get("embedding_dims", {})
        continuous_features = config.get("continuous_features", [])
        continuous_dim = len(continuous_features)

        # Embedding layers
        if vocab_sizes is None:
            vocab_sizes = {}
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for name, dim in embedding_dims.items():
            num_embeddings = vocab_sizes.get(name, 200) + 1  # +1 for padding idx 0
            self.embeddings[name] = nn.Embedding(num_embeddings, dim, padding_idx=0)
            total_emb_dim += dim

        input_dim = continuous_dim + total_emb_dim

        # Recurrent layer
        rnn_cls = nn.LSTM if arch == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        # Primary head: lap time
        self.fc = nn.Linear(hidden_size, 1)

        # Auxiliary head: degradation rate (lap-to-lap delta)
        self.aux_enabled = config.get("aux_deg_head", True)
        if self.aux_enabled:
            self.fc_deg = nn.Linear(hidden_size, 1)
        else:
            self.fc_deg = None

        # Build module dict for param management
        modules = {
            "embeddings": self.embeddings,
            "rnn": self.rnn,
            "dropout": self.dropout,
            "fc": self.fc,
        }
        if self.fc_deg is not None:
            modules["fc_deg"] = self.fc_deg

        self.all_params = nn.ModuleDict(modules)

    def forward(self, continuous, categoricals, lengths):
        torch = _import_torch()

        # Clamp lengths to at least 1 to avoid pack_padded_sequence errors
        lengths = lengths.clamp(min=1)

        # Embed categoricals
        emb_parts = []
        for name, indices in categoricals.items():
            if name in self.embeddings:
                emb_parts.append(self.embeddings[name](indices))

        x = torch.cat([continuous, *emb_parts], dim=-1) if emb_parts else continuous

        # Pack padded sequences
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = self.dropout(output)

        # Primary prediction: lap time
        lap_time_pred = self.fc(output).squeeze(-1)

        # Auxiliary prediction: degradation rate
        if self.aux_enabled and self.fc_deg is not None:
            deg_rate_pred = self.fc_deg(output).squeeze(-1)
        else:
            deg_rate_pred = None

        return lap_time_pred, deg_rate_pred


class SequenceDegradationModel(DegradationModel):
    """LSTM/GRU model with MC dropout for uncertainty estimation."""

    def __init__(self):
        self.net: StintSequenceNet | None = None
        self.config: dict = {}
        self.label_encoders: dict = {}
        self.feature_stats: dict = {}  # For normalization
        self.target_stats: dict = {}  # For target normalization
        self.vocab_sizes: dict[str, int] = {}

    def fit(self, train_df: pd.DataFrame, config: dict) -> None:
        torch = _import_torch()

        self.config = config
        model_config = config.get("model", {})
        training_config = model_config.get("training", {})

        # Build label encoders for categorical features
        embedding_dims = model_config.get("embedding_dims", {})
        for col in embedding_dims:
            if col in train_df.columns:
                unique_vals = sorted(train_df[col].dropna().unique())
                self.label_encoders[col] = {
                    v: i + 1 for i, v in enumerate(unique_vals)
                }  # 0 reserved for padding
                self.vocab_sizes[col] = len(unique_vals)

        # Compute feature normalization stats (handle NaN columns)
        continuous_features = model_config.get("continuous_features", [])
        for col in continuous_features:
            if col in train_df.columns:
                col_data = train_df[col].dropna()
                mean_val = float(col_data.mean()) if len(col_data) > 0 else 0.0
                std_val = float(col_data.std()) if len(col_data) > 1 else 1.0
                self.feature_stats[col] = {
                    "mean": mean_val,
                    "std": std_val if std_val > 0 else 1.0,
                }
            else:
                # Feature not in data — use neutral defaults
                self.feature_stats[col] = {"mean": 0.0, "std": 1.0}

        # Normalize target (lap_time_seconds) for stable training
        if "lap_time_seconds" in train_df.columns:
            target_data = train_df["lap_time_seconds"].dropna()
            self.target_stats = {
                "mean": float(target_data.mean()),
                "std": float(target_data.std()) if len(target_data) > 1 else 1.0,
            }
            if self.target_stats["std"] == 0:
                self.target_stats["std"] = 1.0

        self.net = StintSequenceNet(model_config, vocab_sizes=self.vocab_sizes)

        # Prepare stint-level sequences
        sequences = self._prepare_sequences(train_df, model_config)
        if not sequences:
            logger.warning("No valid sequences found for training")
            return

        # Training loop
        optimizer = torch.optim.AdamW(
            self.net.all_params.parameters(),
            lr=training_config.get("learning_rate", 0.001),
            weight_decay=training_config.get("weight_decay", 0.0001),
        )
        epochs = training_config.get("epochs", 100)
        patience = training_config.get("patience", 15)
        grad_clip = training_config.get("grad_clip", 1.0)
        aux_weight = model_config.get("aux_loss_weight", 0.3)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.net.all_params.train()
            total_loss = 0.0
            n_batches = 0

            for batch in sequences:
                optimizer.zero_grad()
                lap_time_pred, deg_rate_pred = self.net.forward(
                    batch["continuous"],
                    batch["categoricals"],
                    batch["lengths"],
                )

                mask = batch["mask"]
                if not mask.any():
                    continue

                # ── Stint-lap weighted loss ──────────────────────────
                # Weight late-stint laps 2-3x more to learn cliffs
                weights = batch["weights"]  # (batch, seq_len)

                # Primary loss: weighted MSE on lap times
                residuals_sq = (lap_time_pred[mask] - batch["targets"][mask]) ** 2
                w = weights[mask]
                primary_loss = (residuals_sq * w).mean()

                # Auxiliary loss: degradation rate (lap-to-lap delta)
                aux_loss = torch.tensor(0.0)
                if deg_rate_pred is not None and "deltas" in batch:
                    delta_mask = batch["delta_mask"]
                    if delta_mask.any():
                        delta_residuals_sq = (
                            deg_rate_pred[delta_mask] - batch["deltas"][delta_mask]
                        ) ** 2
                        dw = weights[delta_mask]
                        aux_loss = (delta_residuals_sq * dw).mean()

                loss = primary_loss + aux_weight * aux_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Epoch {epoch}: NaN/Inf loss detected, skipping batch")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.all_params.parameters(), grad_clip)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                continue

            avg_loss = total_loss / n_batches

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss={avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.net.all_params.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}, best loss: {best_loss:.6f}")
                    break

        # Restore best model state
        if best_state is not None:
            self.net.all_params.load_state_dict(best_state)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        torch = _import_torch()
        assert self.net is not None
        self.net.all_params.eval()
        model_config = self.config.get("model", {})

        with torch.no_grad():
            preds = self._predict_by_stint(df, model_config, mc_dropout=False)

        return preds

    def predict_interval(
        self, df: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC dropout prediction intervals."""
        torch = _import_torch()
        model_config = self.config.get("model", {})
        mc_config = model_config.get("mc_dropout", {})
        n_samples = mc_config.get("num_samples", 50)

        assert self.net is not None

        all_preds = []
        with torch.no_grad():
            for _i in range(n_samples):
                preds = self._predict_by_stint(df, model_config, mc_dropout=True)
                all_preds.append(preds)

        all_preds_arr = np.stack(all_preds)
        lower = np.percentile(all_preds_arr, 100 * alpha / 2, axis=0)
        upper = np.percentile(all_preds_arr, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    def _predict_by_stint(
        self,
        df: pd.DataFrame,
        model_config: dict,
        mc_dropout: bool = False,
    ) -> np.ndarray:
        """Predict by grouping into stints and reassembling in original order."""
        _import_torch()
        assert self.net is not None

        if mc_dropout:
            self.net.all_params.train()  # Enable dropout
        else:
            self.net.all_params.eval()

        group_cols = ["race_id", "driver_id", "stint_number"]
        available_groups = [c for c in group_cols if c in df.columns]

        # If we can't group, treat as single sequence
        if not available_groups:
            batch = self._df_to_batch(df, model_config)
            lap_time_pred, _ = self.net.forward(
                batch["continuous"], batch["categoricals"], batch["lengths"]
            )
            preds = lap_time_pred.numpy().flatten()[: len(df)]
            return self._denormalize_targets(preds)

        predictions = np.full(len(df), np.nan)

        for _, group in df.groupby(available_groups):
            sort_col = "stint_lap" if "stint_lap" in group.columns else "tyre_life"
            sorted_group = group.sort_values(sort_col)
            original_indices = sorted_group.index

            batch = self._stints_to_batch([sorted_group], model_config)
            lap_time_pred, _ = self.net.forward(
                batch["continuous"], batch["categoricals"], batch["lengths"]
            )
            stint_preds = lap_time_pred.numpy().flatten()[: len(sorted_group)]
            stint_preds = self._denormalize_targets(stint_preds)

            for i, idx in enumerate(original_indices):
                loc = df.index.get_loc(idx)
                if i < len(stint_preds):
                    predictions[loc] = stint_preds[i]

        # Fill any remaining NaN with mean prediction
        nan_mask = np.isnan(predictions)
        if nan_mask.any():
            fallback = self.target_stats.get("mean", 90.0)
            predictions[nan_mask] = fallback
            logger.warning(f"Filled {nan_mask.sum()} NaN predictions with fallback {fallback:.1f}s")

        return predictions

    def _denormalize_targets(self, preds: np.ndarray) -> np.ndarray:
        """Reverse target normalization."""
        if self.target_stats:
            return preds * self.target_stats["std"] + self.target_stats["mean"]
        return preds

    def save(self, path: Path) -> None:
        torch = _import_torch()
        assert self.net is not None
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.all_params.state_dict(), path / "model.pt")
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "label_encoders": self.label_encoders,
                    "feature_stats": self.feature_stats,
                    "target_stats": self.target_stats,
                    "vocab_sizes": self.vocab_sizes,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "SequenceDegradationModel":
        torch = _import_torch()
        instance = cls()
        with open(path / "metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        instance.config = meta["config"]
        instance.label_encoders = meta["label_encoders"]
        instance.feature_stats = meta["feature_stats"]
        instance.target_stats = meta.get("target_stats", {})
        instance.vocab_sizes = meta.get("vocab_sizes", {})

        model_config = instance.config.get("model", {})
        instance.net = StintSequenceNet(model_config, vocab_sizes=instance.vocab_sizes)
        instance.net.all_params.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        return instance

    def _prepare_sequences(self, df: pd.DataFrame, model_config: dict) -> list[dict]:
        """Group laps by (race, driver, stint) and create padded batch sequences."""
        _import_torch()

        group_cols = ["race_id", "driver_id", "stint_number"]
        available_groups = [c for c in group_cols if c in df.columns]
        if not available_groups:
            return [self._df_to_batch(df, model_config)]

        max_len = model_config.get("max_stint_length", 50)
        batch_size = model_config.get("training", {}).get("batch_size", 64)
        batches = []
        current_batch = []

        for _, group in df.groupby(available_groups):
            sort_col = "stint_lap" if "stint_lap" in group.columns else "tyre_life"
            group = group.sort_values(sort_col)
            if len(group) < 2:
                continue
            current_batch.append(group.head(max_len))
            if len(current_batch) >= batch_size:
                batches.append(self._stints_to_batch(current_batch, model_config))
                current_batch = []

        if current_batch:
            batches.append(self._stints_to_batch(current_batch, model_config))

        logger.info(
            f"Prepared {len(batches)} batches from {df['race_id'].nunique() if 'race_id' in df.columns else '?'} races"
        )
        return batches

    def _stints_to_batch(self, stints: list[pd.DataFrame], model_config: dict) -> dict:
        """Convert list of stint DataFrames to a padded batch dict.

        Includes:
        - stint-lap weighted loss weights (2-3x for laps 15+)
        - lap-to-lap delta targets for auxiliary degradation head
        """
        torch = _import_torch()

        continuous_features = model_config.get("continuous_features", [])
        embedding_dims = model_config.get("embedding_dims", {})
        max_len = max(len(s) for s in stints)

        batch_size = len(stints)
        continuous = torch.zeros(batch_size, max_len, len(continuous_features))
        targets = torch.zeros(batch_size, max_len)
        deltas = torch.zeros(batch_size, max_len)  # lap-to-lap delta
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        delta_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        weights = torch.ones(batch_size, max_len)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        categoricals = {
            name: torch.zeros(batch_size, max_len, dtype=torch.long) for name in embedding_dims
        }

        for i, stint in enumerate(stints):
            seq_len = len(stint)
            lengths[i] = seq_len
            mask[i, :seq_len] = True

            # ── Stint-lap weights ────────────────────────────────
            # Ramp from 1.0 at lap 1 to 3.0 at lap 20+
            if "stint_lap" in stint.columns:
                stint_laps = stint["stint_lap"].values.astype(float)
            else:
                stint_laps = np.arange(1, seq_len + 1, dtype=float)

            w = np.where(
                stint_laps >= 20,
                3.0,
                np.where(stint_laps >= 10, 1.0 + 2.0 * (stint_laps - 10) / 10, 1.0),
            )
            weights[i, :seq_len] = torch.tensor(w, dtype=torch.float32)

            # ── Continuous features ──────────────────────────────
            for j, col in enumerate(continuous_features):
                if col in stint.columns:
                    vals = stint[col].fillna(0).values.astype(float)
                    vals = np.where(np.isfinite(vals), vals, 0.0)
                    stats = self.feature_stats.get(col, {"mean": 0, "std": 1})
                    vals = (vals - stats["mean"]) / stats["std"]
                    continuous[i, :seq_len, j] = torch.tensor(vals, dtype=torch.float32)

            # ── Categorical features ─────────────────────────────
            for name in embedding_dims:
                if name in stint.columns:
                    encoder = self.label_encoders.get(name, {})
                    indices = [encoder.get(v, 0) for v in stint[name].values]
                    categoricals[name][i, :seq_len] = torch.tensor(indices, dtype=torch.long)

            # ── Targets: lap time + delta ────────────────────────
            if "lap_time_seconds" in stint.columns:
                target_vals = stint["lap_time_seconds"].fillna(0).values.astype(float)

                # Compute lap-to-lap deltas (normalised)
                if self.target_stats:
                    norm_targets = (target_vals - self.target_stats["mean"]) / self.target_stats[
                        "std"
                    ]
                    targets[i, :seq_len] = torch.tensor(norm_targets, dtype=torch.float32)

                    # Delta = target[t] - target[t-1] in normalised space
                    if seq_len >= 2:
                        delta_vals = np.diff(norm_targets, prepend=norm_targets[0])
                        delta_vals[0] = 0.0  # No delta for first lap
                        deltas[i, :seq_len] = torch.tensor(delta_vals, dtype=torch.float32)
                        delta_mask[i, 1:seq_len] = True  # Skip first lap
                else:
                    targets[i, :seq_len] = torch.tensor(target_vals, dtype=torch.float32)

        return {
            "continuous": continuous,
            "categoricals": categoricals,
            "targets": targets,
            "deltas": deltas,
            "mask": mask,
            "delta_mask": delta_mask,
            "weights": weights,
            "lengths": lengths,
        }

    def _df_to_batch(self, df: pd.DataFrame, model_config: dict) -> dict:
        """Convert a flat DataFrame to a single-sequence batch."""
        return self._stints_to_batch([df], model_config)
