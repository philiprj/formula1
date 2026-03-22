"""Model comparison visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model_comparison_bar(
    results: dict[str, dict],
    metric: str = "mae",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar chart comparing a single metric across models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    names = []
    values = []
    for name, result in sorted(results.items()):
        agg = result.get("aggregate", {})
        if metric in agg:
            names.append(name)
            values.append(agg[metric])

    bars = ax.bar(names, values, color="steelblue", edgecolor="black")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison — {metric.upper()}")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    return fig


def plot_fold_mae_distribution(
    results: dict[str, dict],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Box plot of per-fold MAE distribution across models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    names = []
    data = []
    for name, result in sorted(results.items()):
        fold_results = result.get("fold_results", [])
        maes = [f["mae"] for f in fold_results if "mae" in f]
        if maes:
            names.append(name)
            data.append(maes)

    ax.boxplot(data, labels=names)
    ax.set_ylabel("MAE (seconds)")
    ax.set_title("Per-Fold MAE Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    return fig


def plot_degradation_curves_comparison(
    model_curves: dict[str, pd.DataFrame],
    compound: str = "MEDIUM",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Overlay predicted degradation curves from multiple models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_curves)))

    for (name, curve), color in zip(sorted(model_curves.items()), colors, strict=False):
        laps = curve["lap_in_stint"]
        ax.plot(laps, curve["predicted_lap_time"], label=name, color=color, linewidth=2)
        ax.fill_between(
            laps,
            curve["lower_bound"],
            curve["upper_bound"],
            alpha=0.1,
            color=color,
        )

    ax.set_xlabel("Lap in Stint")
    ax.set_ylabel("Predicted Lap Time (s)")
    ax.set_title(f"Degradation Curve Comparison — {compound}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
