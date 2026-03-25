"""Visualization of tire degradation curves."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1deg.viz.theme import COMPOUND as COMPOUND_COLORS


def plot_degradation_by_compound(
    df: pd.DataFrame,
    race_id: str | None = None,
    driver_id: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot lap time vs tire life, colored by compound.

    Overlays individual stint traces and a smoothed trend line per compound.
    """
    subset = df.copy()
    if race_id:
        subset = subset[subset["race_id"] == race_id]
    if driver_id:
        subset = subset[subset["driver_id"] == driver_id]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    for compound, group in subset.groupby("compound"):
        color = COMPOUND_COLORS.get(compound, "#999999")
        ax.scatter(
            group["tyre_life"],
            group["lap_time_seconds"],
            c=color,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.4,
            s=15,
            label=compound,
        )

        # Rolling mean trend
        sorted_group = group.sort_values("tyre_life")
        if len(sorted_group) >= 5:
            trend = sorted_group.groupby("tyre_life")["lap_time_seconds"].mean()
            ax.plot(trend.index, trend.values, color=color, linewidth=2, alpha=0.8)

    title_parts = ["Tire Degradation"]
    if race_id:
        title_parts.append(f"Race: {race_id}")
    if driver_id:
        title_parts.append(f"Driver: {driver_id}")
    ax.set_title(" — ".join(title_parts))
    ax.set_xlabel("Tire Life (laps)")
    ax.set_ylabel("Lap Time (seconds)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    title: str = "Predicted vs Actual Lap Times",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Scatter plot of predicted vs actual lap times."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    ax.scatter(y_true, y_pred, alpha=0.3, s=5, c="steelblue")

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Perfect")

    ax.set_xlabel("Actual Lap Time (s)")
    ax.set_ylabel("Predicted Lap Time (s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_stint_prediction(
    curve_df: pd.DataFrame,
    actual_times: np.ndarray | None = None,
    compound: str = "",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a predicted degradation curve with confidence band."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    laps = curve_df["lap_in_stint"]
    color = COMPOUND_COLORS.get(compound.upper(), "#4477AA")

    ax.plot(laps, curve_df["predicted_lap_time"], color=color, linewidth=2, label="Predicted")
    ax.plot(laps, curve_df["upper_bound"], color=color, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.plot(
        laps,
        curve_df["lower_bound"],
        color=color,
        linewidth=0.8,
        linestyle=":",
        alpha=0.5,
        label="95% PI",
    )

    if actual_times is not None:
        ax.scatter(
            range(1, len(actual_times) + 1),
            actual_times,
            c="black",
            s=20,
            zorder=5,
            label="Actual",
        )

    # Clamp y-axis to prediction line so PI bounds don't stretch the view
    pred = curve_df["predicted_lap_time"]
    y_margin = (pred.max() - pred.min()) * 0.15 + 0.3
    ax.set_ylim(pred.min() - y_margin, pred.max() + y_margin)

    ax.set_xlabel("Lap in Stint")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title(f"Stint Prediction — {compound}" if compound else "Stint Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
