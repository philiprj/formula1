"""Plotly chart builders for the F1 tyre degradation dashboard.

All charts use the Catppuccin Mocha dark theme from viz/theme.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from f1deg.viz.theme import ACCENT, COMPOUND, GRID, SLATE, SURFACE, TEXT, TEXT_DIM

# ---------------------------------------------------------------------------
# Shared layout helper
# ---------------------------------------------------------------------------


def _base_layout(**overrides) -> go.Layout:
    """Create a dark-themed Plotly layout."""
    defaults = {
        "paper_bgcolor": SLATE,
        "plot_bgcolor": SURFACE,
        "font": {"color": TEXT, "family": "Inter, Helvetica Neue, Arial, sans-serif"},
        "xaxis": {"gridcolor": GRID, "gridwidth": 0.5, "zerolinecolor": GRID},
        "yaxis": {"gridcolor": GRID, "gridwidth": 0.5, "zerolinecolor": GRID},
        "legend": {"bgcolor": SURFACE, "bordercolor": GRID, "borderwidth": 1},
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
        "hovermode": "x unified",
    }
    defaults.update(overrides)
    return go.Layout(**defaults)


def _compound_color(compound: str) -> str:
    return COMPOUND.get(compound.upper(), TEXT_DIM)


# ---------------------------------------------------------------------------
# Degradation curves
# ---------------------------------------------------------------------------


def plot_degradation_curves(
    curves: dict[str, pd.DataFrame],
    title: str = "Tyre Degradation Curves",
) -> go.Figure:
    """Plot degradation curves for multiple compounds with prediction intervals.

    Args:
        curves: {compound_name: DataFrame with columns
                 [lap_in_stint, predicted_lap_time, lower_bound, upper_bound]}
    """
    fig = go.Figure(layout=_base_layout(title=title))

    for compound, df in curves.items():
        color = _compound_color(compound)
        laps = df["lap_in_stint"]

        # Prediction interval boundary lines (thin dashed)
        fig.add_trace(
            go.Scatter(
                x=laps,
                y=df["upper_bound"],
                mode="lines",
                line={"color": _hex_to_rgba(color, 0.4), "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=laps,
                y=df["lower_bound"],
                mode="lines",
                line={"color": _hex_to_rgba(color, 0.4), "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Main prediction line
        fig.add_trace(
            go.Scatter(
                x=laps,
                y=df["predicted_lap_time"],
                mode="lines",
                name=compound,
                line={"color": color, "width": 2.5},
                hovertemplate="%{y:.3f}s<extra>" + compound + "</extra>",
            )
        )

    # Clamp y-axis to prediction lines so PI bounds don't stretch the view
    all_pred = pd.concat([df["predicted_lap_time"] for df in curves.values()])
    y_margin = (all_pred.max() - all_pred.min()) * 0.15 + 0.3
    fig.update_layout(
        xaxis_title="Lap in Stint",
        yaxis_title="Predicted Lap Time (s)",
        yaxis_range=[all_pred.min() - y_margin, all_pred.max() + y_margin],
    )
    return fig


def plot_degradation_animated(
    curves: dict[str, pd.DataFrame],
    title: str = "Tyre Degradation (Animated)",
) -> go.Figure:
    """Animated degradation curves that build up lap-by-lap."""
    if not curves:
        return go.Figure(layout=_base_layout(title=title))

    max_laps = max(len(df) for df in curves.values())
    compounds = list(curves.keys())

    # Initial frame: just the first lap
    fig = go.Figure(layout=_base_layout(title=title))

    for compound in compounds:
        color = _compound_color(compound)
        df = curves[compound]
        fig.add_trace(
            go.Scatter(
                x=[df["lap_in_stint"].iloc[0]],
                y=[df["predicted_lap_time"].iloc[0]],
                mode="lines+markers",
                name=compound,
                line={"color": color, "width": 2.5},
                marker={"size": 4},
            )
        )

    # Build frames
    frames = []
    for k in range(1, max_laps + 1):
        frame_data = []
        for compound in compounds:
            df = curves[compound]
            n = min(k, len(df))
            frame_data.append(
                go.Scatter(
                    x=df["lap_in_stint"].iloc[:n],
                    y=df["predicted_lap_time"].iloc[:n],
                    mode="lines+markers",
                    marker={"size": 4},
                )
            )
        frames.append(go.Frame(data=frame_data, name=str(k)))

    fig.frames = frames

    # Determine y-axis range from all data
    all_times = pd.concat([df["predicted_lap_time"] for df in curves.values()])
    y_min = all_times.min() - 0.5
    y_max = all_times.max() + 0.5

    fig.update_layout(
        xaxis={"range": [0.5, max_laps + 0.5], "title": "Lap in Stint"},
        yaxis={"range": [y_min, y_max], "title": "Predicted Lap Time (s)"},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 150, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 50},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": [
                    {
                        "args": [
                            [str(k)],
                            {
                                "frame": {"duration": 150, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 50},
                            },
                        ],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k in range(1, max_laps + 1)
                ],
                "x": 0.05,
                "len": 0.9,
                "currentvalue": {"prefix": "Lap: ", "font": {"color": TEXT}},
                "font": {"color": TEXT_DIM},
                "bgcolor": SURFACE,
                "bordercolor": GRID,
                "activebgcolor": ACCENT[0],
            }
        ],
    )
    return fig


def plot_lap_deltas(
    curves: dict[str, pd.DataFrame],
    title: str = "Lap-over-Lap Degradation Rate",
) -> go.Figure:
    """Bar chart of per-lap time delta for each compound."""
    fig = go.Figure(layout=_base_layout(title=title))

    for compound, df in curves.items():
        color = _compound_color(compound)
        times = df["predicted_lap_time"].values
        deltas = np.diff(times)
        laps = df["lap_in_stint"].values[1:]

        fig.add_trace(
            go.Bar(
                x=laps,
                y=deltas,
                name=compound,
                marker_color=color,
                opacity=0.8,
                hovertemplate="+%{y:.3f}s<extra>" + compound + "</extra>",
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="Lap in Stint",
        yaxis_title="Time Gained (s)",
    )
    return fig


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def plot_model_overlay(
    model_curves: dict[str, pd.DataFrame],
    compound: str,
    title: str | None = None,
) -> go.Figure:
    """Overlay multiple models' curves for a single compound."""
    if title is None:
        title = f"Model Comparison — {compound}"

    fig = go.Figure(layout=_base_layout(title=title))

    for i, (model_name, df) in enumerate(model_curves.items()):
        color = ACCENT[i % len(ACCENT)]
        laps = df["lap_in_stint"]

        # PI boundary lines (thin dashed)
        fig.add_trace(
            go.Scatter(
                x=laps,
                y=df["upper_bound"],
                mode="lines",
                line={"color": _hex_to_rgba(color, 0.35), "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=laps,
                y=df["lower_bound"],
                mode="lines",
                line={"color": _hex_to_rgba(color, 0.35), "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Line
        fig.add_trace(
            go.Scatter(
                x=laps,
                y=df["predicted_lap_time"],
                mode="lines",
                name=model_name,
                line={"color": color, "width": 2.5},
                hovertemplate="%{y:.3f}s<extra>" + model_name + "</extra>",
            )
        )

    # Clamp y-axis to prediction lines so PI bounds don't stretch the view
    all_pred = pd.concat([df["predicted_lap_time"] for df in model_curves.values()])
    y_margin = (all_pred.max() - all_pred.min()) * 0.15 + 0.3
    fig.update_layout(
        xaxis_title="Lap in Stint",
        yaxis_title="Predicted Lap Time (s)",
        yaxis_range=[all_pred.min() - y_margin, all_pred.max() + y_margin],
    )
    return fig


def plot_metrics_bar(
    results: dict[str, dict[str, float]],
    metric: str,
    title: str | None = None,
) -> go.Figure:
    """Bar chart comparing a single metric across models."""
    if title is None:
        title = metric.upper().replace("_", " ")

    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]

    fig = go.Figure(
        data=[
            go.Bar(
                x=models,
                y=values,
                marker_color=ACCENT[: len(models)],
                text=[f"{v:.4f}" for v in values],
                textposition="auto",
                textfont={"color": TEXT},
            )
        ],
        layout=_base_layout(title=title),
    )
    fig.update_layout(yaxis_title=metric.upper().replace("_", " "))
    return fig


# ---------------------------------------------------------------------------
# Strategy simulator
# ---------------------------------------------------------------------------


def plot_race_timeline(
    stint_curves: list[dict],
    pit_laps: list[int],
    title: str = "Race Lap Times",
) -> go.Figure:
    """Plot full race timeline with compound-colored stint segments.

    Args:
        stint_curves: List of dicts with keys: compound, laps (list[int]),
                      times (list[float]), lower (list[float]), upper (list[float])
        pit_laps: Lap numbers where pit stops occur.
    """
    fig = go.Figure(layout=_base_layout(title=title))

    for stint in stint_curves:
        color = _compound_color(stint["compound"])
        laps = stint["laps"]

        # PI boundary lines (thin dashed)
        fig.add_trace(
            go.Scatter(
                x=laps,
                y=stint["upper"],
                mode="lines",
                line={"color": _hex_to_rgba(color, 0.4), "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=laps,
                y=stint["lower"],
                mode="lines",
                line={"color": _hex_to_rgba(color, 0.4), "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Lap time line
        fig.add_trace(
            go.Scatter(
                x=laps,
                y=stint["times"],
                mode="lines",
                name=stint["compound"],
                line={"color": color, "width": 2.5},
                hovertemplate="Lap %{x}: %{y:.3f}s<extra>" + stint["compound"] + "</extra>",
            )
        )

    # Pit stop markers
    for pit_lap in pit_laps:
        fig.add_vline(
            x=pit_lap + 0.5,
            line_dash="dash",
            line_color=TEXT_DIM,
            annotation_text="PIT",
            annotation_font_color=TEXT_DIM,
        )

    # Clamp y-axis to prediction lines so PI bounds don't stretch the view
    all_times = [t for stint in stint_curves for t in stint["times"]]
    if all_times:
        y_min, y_max = min(all_times), max(all_times)
        y_margin = (y_max - y_min) * 0.15 + 0.3
        fig.update_layout(
            xaxis_title="Race Lap",
            yaxis_title="Lap Time (s)",
            yaxis_range=[y_min - y_margin, y_max + y_margin],
        )
    else:
        fig.update_layout(
            xaxis_title="Race Lap",
            yaxis_title="Lap Time (s)",
        )
    return fig


def plot_cumulative_time(
    stint_curves: list[dict],
    pit_times: list[float],
    title: str = "Cumulative Race Time",
) -> go.Figure:
    """Running total race time across all stints."""
    fig = go.Figure(layout=_base_layout(title=title))

    all_laps = []
    all_cum = []
    running = 0.0
    pit_idx = 0

    for i, stint in enumerate(stint_curves):
        if i > 0 and pit_idx < len(pit_times):
            running += pit_times[pit_idx]
            pit_idx += 1

        for lap, t in zip(stint["laps"], stint["times"], strict=False):
            running += t
            all_laps.append(lap)
            all_cum.append(running)

    fig.add_trace(
        go.Scatter(
            x=all_laps,
            y=[c / 60 for c in all_cum],  # convert to minutes
            mode="lines",
            name="Cumulative",
            line={"color": ACCENT[0], "width": 2.5},
            hovertemplate="Lap %{x}: %{y:.1f} min<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Race Lap",
        yaxis_title="Cumulative Time (min)",
    )
    return fig


def plot_strategy_comparison(
    strategies: list[dict],
    title: str = "Strategy Comparison",
) -> go.Figure:
    """Bar chart comparing total race times for saved strategies.

    Args:
        strategies: List of dicts with keys: name, total_time (seconds)
    """
    if not strategies:
        return go.Figure(layout=_base_layout(title=title))

    names = [s["name"] for s in strategies]
    times = [s["total_time"] / 60 for s in strategies]

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=times,
                marker_color=ACCENT[: len(names)],
                text=[f"{t:.1f} min" for t in times],
                textposition="auto",
                textfont={"color": TEXT},
            )
        ],
        layout=_base_layout(title=title),
    )
    fig.update_layout(yaxis_title="Total Race Time (min)")
    return fig


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def plot_residuals_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals vs Predicted",
) -> go.Figure:
    """Scatter plot of residuals against predicted values."""
    residuals = y_true - y_pred

    fig = go.Figure(
        data=[
            go.Scattergl(
                x=y_pred,
                y=residuals,
                mode="markers",
                marker={"color": ACCENT[1], "size": 3, "opacity": 0.5},
                hovertemplate="Predicted: %{x:.2f}s<br>Residual: %{y:.3f}s<extra></extra>",
            )
        ],
        layout=_base_layout(title=title),
    )
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_DIM)
    fig.update_layout(
        xaxis_title="Predicted Lap Time (s)",
        yaxis_title="Residual (s)",
    )
    return fig


def plot_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Distribution",
) -> go.Figure:
    """Histogram of residuals with mean/std annotation."""
    residuals = y_true - y_pred
    mean_r = float(np.mean(residuals))
    std_r = float(np.std(residuals))

    fig = go.Figure(
        data=[
            go.Histogram(
                x=residuals,
                nbinsx=80,
                marker_color=ACCENT[1],
                opacity=0.85,
            )
        ],
        layout=_base_layout(title=title),
    )
    fig.add_vline(x=mean_r, line_dash="dash", line_color=ACCENT[0])
    fig.add_annotation(
        x=mean_r + std_r,
        y=0.95,
        yref="paper",
        text=f"Mean: {mean_r:.3f}s | Std: {std_r:.3f}s",
        showarrow=False,
        font={"color": TEXT, "size": 12},
        bgcolor=SURFACE,
    )
    fig.update_layout(xaxis_title="Residual (s)", yaxis_title="Count")
    return fig


def plot_residuals_by_group(
    residuals: np.ndarray,
    groups: pd.Series,
    group_label: str = "Group",
    title: str | None = None,
) -> go.Figure:
    """Box plot of residuals grouped by a categorical variable."""
    if title is None:
        title = f"Residuals by {group_label}"

    unique_groups = sorted(groups.unique())
    fig = go.Figure(layout=_base_layout(title=title))

    for i, group in enumerate(unique_groups):
        mask = groups == group
        color = COMPOUND.get(str(group).upper(), ACCENT[i % len(ACCENT)])
        fig.add_trace(
            go.Box(
                y=residuals[mask],
                name=str(group),
                marker_color=color,
                line_color=color,
            )
        )

    fig.update_layout(
        xaxis_title=group_label,
        yaxis_title="Residual (s)",
        showlegend=False,
    )
    return fig


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
) -> go.Figure:
    """Scatter plot with identity line."""
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=y_true,
                y=y_pred,
                mode="markers",
                marker={"color": ACCENT[1], "size": 3, "opacity": 0.5},
                hovertemplate="Actual: %{x:.2f}s<br>Predicted: %{y:.2f}s<extra></extra>",
            )
        ],
        layout=_base_layout(title=title),
    )

    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"color": TEXT_DIM, "dash": "dash", "width": 1},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis_title="Actual Lap Time (s)",
        yaxis_title="Predicted Lap Time (s)",
    )
    return fig


# ---------------------------------------------------------------------------
# Data explorer
# ---------------------------------------------------------------------------


def plot_data_scatter(
    df: pd.DataFrame,
    title: str = "Lap Time vs Tyre Life",
) -> go.Figure:
    """Scatter plot of lap times colored by compound."""
    fig = go.Figure(layout=_base_layout(title=title))

    for compound in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        sub = df[df["compound"] == compound]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=sub["tyre_life"],
                y=sub["lap_time_seconds"],
                mode="markers",
                name=compound,
                marker={"color": _compound_color(compound), "size": 3, "opacity": 0.4},
                hovertemplate=f"{compound}<br>Life: %{{x}} laps<br>Time: %{{y:.2f}}s<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Tyre Life (laps)",
        yaxis_title="Lap Time (s)",
    )
    return fig


def plot_lap_time_distributions(
    df: pd.DataFrame,
    title: str = "Lap Time Distribution by Compound",
) -> go.Figure:
    """Overlaid histograms of lap times per compound."""
    fig = go.Figure(layout=_base_layout(title=title))

    for compound in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        sub = df[df["compound"] == compound]
        if sub.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=sub["lap_time_seconds"],
                name=compound,
                marker_color=_compound_color(compound),
                opacity=0.6,
                nbinsx=60,
            )
        )

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Lap Time (s)",
        yaxis_title="Count",
    )
    return fig


def plot_weather_scatter(
    df: pd.DataFrame,
    title: str = "Weather Conditions in Training Data",
) -> go.Figure:
    """Scatter of track temp vs air temp colored by circuit."""
    if "track_temp" not in df.columns or "air_temp" not in df.columns:
        fig = go.Figure(layout=_base_layout(title=title))
        fig.add_annotation(
            text="Weather data not available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"color": TEXT_DIM, "size": 16},
        )
        return fig

    fig = go.Figure(layout=_base_layout(title=title))

    circuits = sorted(df["circuit_id"].unique()) if "circuit_id" in df.columns else []
    for i, circuit in enumerate(circuits[:12]):  # limit to 12 for readability
        sub = df[df["circuit_id"] == circuit]
        fig.add_trace(
            go.Scattergl(
                x=sub["air_temp"],
                y=sub["track_temp"],
                mode="markers",
                name=circuit,
                marker={"color": ACCENT[i % len(ACCENT)], "size": 4, "opacity": 0.5},
            )
        )

    fig.update_layout(
        xaxis_title="Air Temperature (C)",
        yaxis_title="Track Temperature (C)",
    )
    return fig


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
