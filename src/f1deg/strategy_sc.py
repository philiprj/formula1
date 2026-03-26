"""Circuit-specific safety car probability model.

Replaces the flat 4% SC probability with per-circuit rates derived from
historical data. Uses a simple empirical rate with Bayesian smoothing
(beta-binomial) to handle circuits with few races.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Global prior: ~4% SC probability per lap (informative prior from historical average)
PRIOR_ALPHA = 2.0  # pseudo-counts for SC laps
PRIOR_BETA = 48.0  # pseudo-counts for non-SC laps (gives ~4% prior mean)


def compute_sc_rates_per_circuit(df: pd.DataFrame) -> pd.DataFrame:
    """Compute empirical SC probability per lap for each circuit.

    Uses the TrackStatus column to detect SC periods and computes the
    fraction of laps that have an active SC. Applies beta-binomial
    smoothing so circuits with few races don't get extreme estimates.

    Args:
        df: Processed laps DataFrame with 'circuit_id', 'race_id',
            'lap_number', and SC indicator columns.

    Returns:
        DataFrame with columns: circuit_id, total_races, total_laps,
        sc_laps, raw_rate, smoothed_rate.
    """
    records = []

    for circuit_id, circuit_df in df.groupby("circuit_id"):
        total_laps = len(circuit_df)
        total_races = circuit_df["race_id"].nunique()

        # Count laps with active safety car
        sc_laps = 0
        if "laps_since_sc_end" in circuit_df.columns:
            # laps_since_sc_end == 0 means SC is active or just ended this lap
            sc_laps = int((circuit_df["laps_since_sc_end"] == 0).sum())
        elif "TrackStatus" in circuit_df.columns:
            # TrackStatus contains SC codes
            sc_laps = int(circuit_df["TrackStatus"].isin({"4", "6", "7"}).sum())
        elif "had_sc_this_stint" in circuit_df.columns:
            # Approximate: count stints with SC
            sc_stints = circuit_df.groupby(["race_id", "driver_id", "stint_number"])[
                "had_sc_this_stint"
            ].any()
            sc_laps = int(
                circuit_df.merge(
                    sc_stints.reset_index().rename(columns={"had_sc_this_stint": "_had_sc"}),
                    on=["race_id", "driver_id", "stint_number"],
                )["_had_sc"].sum()
                / max(1, circuit_df["driver_id"].nunique())  # per-driver to per-race
            )

        raw_rate = sc_laps / total_laps if total_laps > 0 else 0.0

        # Beta-binomial smoothing
        smoothed_rate = (PRIOR_ALPHA + sc_laps) / (PRIOR_ALPHA + PRIOR_BETA + total_laps)

        records.append(
            {
                "circuit_id": circuit_id,
                "total_races": total_races,
                "total_laps": total_laps,
                "sc_laps": sc_laps,
                "raw_rate": raw_rate,
                "smoothed_rate": smoothed_rate,
            }
        )

    result = pd.DataFrame(records).sort_values("smoothed_rate", ascending=False)
    logger.info(
        f"SC rates: min={result['smoothed_rate'].min():.3f}, "
        f"max={result['smoothed_rate'].max():.3f}, "
        f"mean={result['smoothed_rate'].mean():.3f}"
    )
    return result


def get_sc_probability(
    circuit_id: str,
    sc_rates: pd.DataFrame | None = None,
    df: pd.DataFrame | None = None,
) -> float:
    """Get the SC probability per lap for a circuit.

    If sc_rates is pre-computed, looks up the circuit. Otherwise computes
    from the raw DataFrame. Falls back to the global prior (4%) if the
    circuit is unknown.

    Args:
        circuit_id: Circuit identifier (e.g. "British Grand Prix").
        sc_rates: Pre-computed SC rates from compute_sc_rates_per_circuit().
        df: Raw laps DataFrame (used if sc_rates not provided).

    Returns:
        SC probability per lap (float between 0 and 1).
    """
    if sc_rates is None and df is not None:
        sc_rates = compute_sc_rates_per_circuit(df)

    if sc_rates is not None:
        match = sc_rates[sc_rates["circuit_id"] == circuit_id]
        if len(match) > 0:
            return float(match.iloc[0]["smoothed_rate"])

    # Fallback to prior
    return PRIOR_ALPHA / (PRIOR_ALPHA + PRIOR_BETA)


def print_sc_summary(sc_rates: pd.DataFrame) -> None:
    """Print a formatted summary of SC rates by circuit."""
    print(f"\n{'circuit':<35} {'races':>5} {'SC laps':>7} {'raw':>6} {'smoothed':>8}")
    print("-" * 65)
    for _, row in sc_rates.iterrows():
        print(
            f"{row['circuit_id']:<35} {int(row['total_races']):>5} "
            f"{int(row['sc_laps']):>7} "
            f"{row['raw_rate']:>5.1%} "
            f"{row['smoothed_rate']:>7.1%}"
        )
