"""Model diagnostic utilities for comparing residual patterns.

Helps identify where the Bayesian model underperforms vs GBM and what
interactions or feature groups drive the gap. Guides improvements to
the Bayesian model's priors and structure.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    model_name: str = "model",
) -> pd.DataFrame:
    """Compute residuals and group by key dimensions.

    Returns a DataFrame with per-group MAE broken down by:
    - compound
    - circuit_id
    - stint phase (early/mid/late)
    - SC proximity
    """
    temp = df.copy()
    temp["residual"] = y_true - y_pred
    temp["abs_residual"] = np.abs(temp["residual"])
    temp["model"] = model_name

    summaries = []

    # By compound
    if "compound" in temp.columns:
        for compound, group in temp.groupby("compound"):
            summaries.append(
                {
                    "model": model_name,
                    "dimension": "compound",
                    "group": compound,
                    "mae": group["abs_residual"].mean(),
                    "bias": group["residual"].mean(),
                    "n_laps": len(group),
                }
            )

    # By circuit
    if "circuit_id" in temp.columns:
        for circuit, group in temp.groupby("circuit_id"):
            summaries.append(
                {
                    "model": model_name,
                    "dimension": "circuit_id",
                    "group": str(circuit),
                    "mae": group["abs_residual"].mean(),
                    "bias": group["residual"].mean(),
                    "n_laps": len(group),
                }
            )

    # By stint phase
    if "stint_lap" in temp.columns:
        for phase, mask in [
            ("early_1-5", temp["stint_lap"] <= 5),
            ("mid_6-19", (temp["stint_lap"] > 5) & (temp["stint_lap"] <= 19)),
            ("late_20+", temp["stint_lap"] >= 20),
        ]:
            group = temp[mask]
            if len(group) > 0:
                summaries.append(
                    {
                        "model": model_name,
                        "dimension": "stint_phase",
                        "group": phase,
                        "mae": group["abs_residual"].mean(),
                        "bias": group["residual"].mean(),
                        "n_laps": len(group),
                    }
                )

    # By SC proximity
    if "laps_since_sc_end" in temp.columns:
        for phase, mask in [
            ("post_sc_0-3", temp["laps_since_sc_end"] <= 3),
            ("normal_4+", temp["laps_since_sc_end"] > 3),
        ]:
            group = temp[mask]
            if len(group) > 0:
                summaries.append(
                    {
                        "model": model_name,
                        "dimension": "sc_proximity",
                        "group": phase,
                        "mae": group["abs_residual"].mean(),
                        "bias": group["residual"].mean(),
                        "n_laps": len(group),
                    }
                )

    return pd.DataFrame(summaries)


def compare_model_residuals(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare residual patterns across multiple models.

    Args:
        results: Dict mapping model_name -> (y_true, y_pred).
        df: Feature DataFrame (same for all models).

    Returns:
        Combined DataFrame with per-group MAE for each model,
        plus a "gap" column showing where one model beats another.
    """
    all_summaries = []
    for model_name, (y_true, y_pred) in results.items():
        summary = residual_analysis(y_true, y_pred, df, model_name)
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)

    # Add gap analysis if exactly 2 models
    model_names = list(results.keys())
    if len(model_names) == 2:
        m1, m2 = model_names
        pivot = combined.pivot_table(
            index=["dimension", "group"],
            columns="model",
            values="mae",
        ).reset_index()
        if m1 in pivot.columns and m2 in pivot.columns:
            pivot["mae_gap"] = pivot[m1] - pivot[m2]
            pivot["better_model"] = np.where(pivot["mae_gap"] > 0, m2, m1)
            logger.info(f"\nResidual gap analysis ({m1} vs {m2}):")
            for _, row in pivot.iterrows():
                gap = row["mae_gap"]
                better = row["better_model"]
                logger.info(
                    f"  {row['dimension']}/{row['group']}: {better} wins by {abs(gap):.3f}s"
                )

    return combined


def identify_bayesian_improvement_areas(
    bayesian_residuals: pd.DataFrame,
    gbm_residuals: pd.DataFrame,
) -> list[dict]:
    """Identify specific areas where Bayesian model should improve.

    Compares Bayesian vs GBM residual patterns and returns ranked
    suggestions for prior/structure improvements.
    """
    merged = bayesian_residuals.merge(
        gbm_residuals,
        on=["dimension", "group"],
        suffixes=("_bayesian", "_gbm"),
    )

    if merged.empty:
        return []

    merged["gap"] = merged["mae_bayesian"] - merged["mae_gbm"]
    merged = merged.sort_values("gap", ascending=False)

    suggestions = []
    for _, row in merged.head(10).iterrows():
        if row["gap"] <= 0:
            continue
        suggestions.append(
            {
                "dimension": row["dimension"],
                "group": row["group"],
                "bayesian_mae": round(row["mae_bayesian"], 3),
                "gbm_mae": round(row["mae_gbm"], 3),
                "gap": round(row["gap"], 3),
                "bayesian_bias": round(row.get("bias_bayesian", 0), 3),
                "n_laps": int(row.get("n_laps_bayesian", 0)),
                "suggestion": _suggest_fix(row),
            }
        )

    return suggestions


def _suggest_fix(row: pd.Series) -> str:
    """Generate a specific suggestion based on the residual pattern."""
    dim = row["dimension"]
    group = row["group"]
    bias = row.get("bias_bayesian", 0)

    if dim == "compound" and group in ("SOFT", "INTERMEDIATE", "WET"):
        if bias > 0:
            return f"Bayesian model under-predicts {group} lap times — consider tighter/higher deg_rate prior for {group}"
        return f"Bayesian model over-predicts {group} — deg_rate prior may be too aggressive"

    if dim == "stint_phase" and "late" in group:
        return "Late-stint degradation (cliff behavior) poorly captured — consider non-linear degradation function or compound-specific cliff priors"

    if dim == "circuit_id":
        if bias > 0:
            return f"Circuit {group}: under-prediction suggests circuit_offset prior is too low or circuit-specific deg_rate needs adjustment"
        return f"Circuit {group}: over-prediction suggests tightening circuit-level partial pooling"

    if dim == "sc_proximity" and "post_sc" in group:
        return "Post-SC tire warm-up behavior not well captured — consider adding SC restart latent state to the state-space model"

    return "Investigate feature interactions that GBM captures but the Bayesian linear structure misses"
