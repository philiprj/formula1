"""Tier 4: LSTM/GRU sequence model for tire degradation.

Operates over stint-level sequences with MC dropout for uncertainty estimation.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from f1deg.models.base import DegradationModel

# Lazy import for optional torch dependency
_torch = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


class StintSequenceNet:
    """PyTorch LSTM/GRU network for stint-level lap time prediction."""

    def __init__(self, config: dict):
        _import_torch()
        import torch.nn as nn

        self.config = config
        arch = config.get("architecture", "lstm")
        hidden_size = config.get("hidden_size", 128)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.2)

        embedding_dims = config.get("embedding_dims", {})
        continuous_features = config.get("continuous_features", [])
        continuous_dim = len(continuous_features)

        # Embedding layers
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for name, dim in embedding_dims.items():
            # Max vocab sizes (will be set during fit)
            self.embeddings[name] = nn.Embedding(200, dim)
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
        self.fc = nn.Linear(hidden_size, 1)

        # Combine into a module for easy save/load
        self.all_params = nn.ModuleDict(
            {
                "embeddings": self.embeddings,
                "rnn": self.rnn,
                "dropout": self.dropout,
                "fc": self.fc,
            }
        )

    def forward(self, continuous, categoricals, lengths):
        torch = _import_torch()

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
        predictions = self.fc(output).squeeze(-1)
        return predictions


class SequenceDegradationModel(DegradationModel):
    """LSTM/GRU model with MC dropout for uncertainty estimation."""

    def __init__(self):
        self.net: StintSequenceNet | None = None
        self.config: dict = {}
        self.label_encoders: dict = {}
        self.feature_stats: dict = {}  # For normalization

    def fit(self, train_df: pd.DataFrame, config: dict) -> None:
        torch = _import_torch()
        import torch.nn as nn

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

        # Compute feature normalization stats
        continuous_features = model_config.get("continuous_features", [])
        for col in continuous_features:
            if col in train_df.columns:
                self.feature_stats[col] = {
                    "mean": float(train_df[col].mean()),
                    "std": float(train_df[col].std()) or 1.0,
                }

        self.net = StintSequenceNet(model_config)

        # Prepare stint-level sequences
        sequences = self._prepare_sequences(train_df, model_config)

        # Training loop
        optimizer = torch.optim.AdamW(
            self.net.all_params.parameters(),
            lr=training_config.get("learning_rate", 0.001),
            weight_decay=training_config.get("weight_decay", 0.0001),
        )
        criterion = nn.MSELoss()
        epochs = training_config.get("epochs", 100)
        patience = training_config.get("patience", 10)

        best_loss = float("inf")
        patience_counter = 0

        for _epoch in range(epochs):
            self.net.all_params.train()
            total_loss = 0.0

            for batch in sequences:
                optimizer.zero_grad()
                predictions = self.net.forward(
                    batch["continuous"],
                    batch["categoricals"],
                    batch["lengths"],
                )

                # Mask padded positions
                mask = batch["mask"]
                loss = criterion(predictions[mask], batch["targets"][mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(sequences), 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        torch = _import_torch()
        assert self.net is not None
        self.net.all_params.eval()
        model_config = self.config.get("model", {})

        with torch.no_grad():
            batch = self._df_to_batch(df, model_config)
            predictions = self.net.forward(
                batch["continuous"],
                batch["categoricals"],
                batch["lengths"],
            )
        return np.asarray(predictions.numpy().flatten()[: len(df)])

    def predict_interval(
        self, df: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC dropout prediction intervals."""
        torch = _import_torch()
        model_config = self.config.get("model", {})
        mc_config = model_config.get("mc_dropout", {})
        n_samples = mc_config.get("num_samples", 100)

        assert self.net is not None
        self.net.all_params.train()  # Enable dropout

        all_preds = []
        with torch.no_grad():
            batch = self._df_to_batch(df, model_config)
            for _ in range(n_samples):
                pred = self.net.forward(
                    batch["continuous"],
                    batch["categoricals"],
                    batch["lengths"],
                )
                all_preds.append(pred.numpy().flatten()[: len(df)])

        all_preds_arr = np.stack(all_preds)
        lower = np.percentile(all_preds_arr, 100 * alpha / 2, axis=0)
        upper = np.percentile(all_preds_arr, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

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

        model_config = instance.config.get("model", {})
        instance.net = StintSequenceNet(model_config)
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
            group = group.sort_values("stint_lap" if "stint_lap" in group.columns else "tyre_life")
            if len(group) < 2:
                continue
            current_batch.append(group.head(max_len))
            if len(current_batch) >= batch_size:
                batches.append(self._stints_to_batch(current_batch, model_config))
                current_batch = []

        if current_batch:
            batches.append(self._stints_to_batch(current_batch, model_config))

        return batches

    def _stints_to_batch(self, stints: list[pd.DataFrame], model_config: dict) -> dict:
        """Convert list of stint DataFrames to a padded batch dict."""
        torch = _import_torch()

        continuous_features = model_config.get("continuous_features", [])
        embedding_dims = model_config.get("embedding_dims", {})
        max_len = max(len(s) for s in stints)

        batch_size = len(stints)
        continuous = torch.zeros(batch_size, max_len, len(continuous_features))
        targets = torch.zeros(batch_size, max_len)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        categoricals = {
            name: torch.zeros(batch_size, max_len, dtype=torch.long) for name in embedding_dims
        }

        for i, stint in enumerate(stints):
            seq_len = len(stint)
            lengths[i] = seq_len
            mask[i, :seq_len] = True

            for j, col in enumerate(continuous_features):
                if col in stint.columns:
                    vals = stint[col].values.astype(float)
                    stats = self.feature_stats.get(col, {"mean": 0, "std": 1})
                    vals = (vals - stats["mean"]) / stats["std"]
                    continuous[i, :seq_len, j] = torch.tensor(vals, dtype=torch.float32)

            for name in embedding_dims:
                if name in stint.columns:
                    encoder = self.label_encoders.get(name, {})
                    indices = [encoder.get(v, 0) for v in stint[name].values]
                    categoricals[name][i, :seq_len] = torch.tensor(indices, dtype=torch.long)

            if "lap_time_seconds" in stint.columns:
                targets[i, :seq_len] = torch.tensor(
                    stint["lap_time_seconds"].values, dtype=torch.float32
                )

        return {
            "continuous": continuous,
            "categoricals": categoricals,
            "targets": targets,
            "mask": mask,
            "lengths": lengths,
        }

    def _df_to_batch(self, df: pd.DataFrame, model_config: dict) -> dict:
        """Convert a flat DataFrame to a single-sequence batch."""
        return self._stints_to_batch([df], model_config)
