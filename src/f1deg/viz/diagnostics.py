"""Diagnostic plots for model evaluation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot residuals (actual - predicted) vs predicted."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.3, s=5, c="steelblue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Lap Time (s)")
    ax.set_ylabel("Residual (s)")
    ax.set_title("Residual Plot")
    ax.grid(True, alpha=0.3)

    return fig


def plot_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 50,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Histogram of residuals."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    residuals = y_true - y_pred
    ax.hist(residuals, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_xlabel("Residual (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals — Mean: {np.mean(residuals):.3f}s, Std: {np.std(residuals):.3f}s")
    ax.grid(True, alpha=0.3)

    return fig


def plot_residuals_by_group(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    group_col: str = "compound",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Box plot of residuals grouped by a categorical variable."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    plot_df = df.copy()
    plot_df["residual"] = df["lap_time_seconds"].values - y_pred

    groups = sorted(plot_df[group_col].unique())
    data = [plot_df[plot_df[group_col] == g]["residual"].values for g in groups]

    ax.boxplot(data, labels=groups, showfliers=False)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel(group_col)
    ax.set_ylabel("Residual (s)")
    ax.set_title(f"Residuals by {group_col}")
    ax.grid(True, alpha=0.3, axis="y")

    return fig
