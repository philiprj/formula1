"""Undercut dynamics model for race strategy.

Quantifies the fresh-tyre advantage when pitting ahead of a rival.
The "undercut" is the gain from running on fresh tyres while your
competitor stays out on worn rubber. This is the primary driver of
pit stop timing in modern F1 — not absolute tyre degradation.

The model uses empirical lookup tables derived from historical race data,
parameterised by:
    - Opponent tyre age (how worn their tyres are)
    - New compound (what you're switching to)
    - Circuit (some circuits have stronger undercuts)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Empirical pit timing (from 2022-2025 race data, 1502 pit stops) ──────
# Median first stint length by starting compound and inter-quartile range.
# Used as a prior on the optimizer to prevent recommending absurdly late pits.
EMPIRICAL_PIT_TIMING = {
    "SOFT": {"median": 12, "std": 5, "p25": 9, "p75": 17},
    "MEDIUM": {"median": 15, "std": 6, "p25": 10, "p75": 22},
    "HARD": {"median": 26, "std": 7, "p25": 18, "p75": 31},
}


# ── Default undercut lookup tables ───────────────────────────────────────
# Derived from 2022-2025 race data (20K+ observations).
# Key: opponent tyre age bucket → fresh tyre advantage (s/lap, first 5 laps)

DEFAULT_UNDERCUT_BY_AGE = {
    # (age_lo, age_hi): advantage_per_lap
    (5, 10): 0.25,
    (10, 15): 0.45,
    (15, 20): 0.73,
    (20, 25): 0.84,
    (25, 30): 0.93,
    (30, 35): 1.10,
    (35, 99): 1.32,
}

# Compound adjustment (relative to HARD baseline)
COMPOUND_FACTOR = {
    "SOFT": 0.90,  # Softs degrade fast themselves, so advantage fades quicker
    "MEDIUM": 1.06,  # Best undercut compound
    "HARD": 1.00,  # Baseline
}

# Number of laps the fresh-tyre advantage persists
UNDERCUT_WINDOW_LAPS = 5


def estimate_undercut_advantage(
    opponent_tyre_age: int,
    new_compound: str = "MEDIUM",
    circuit_factor: float = 1.0,
) -> float:
    """Estimate the per-lap fresh tyre advantage for an undercut.

    Args:
        opponent_tyre_age: How many laps the opponent has done on their tyres.
        new_compound: The compound you're switching to.
        circuit_factor: Circuit-specific multiplier (>1 = stronger undercut).

    Returns:
        Estimated advantage in seconds per lap for the first 5 laps after pitting.
    """
    # Lookup base advantage by opponent tyre age
    base_advantage = 0.25  # minimum
    for (lo, hi), adv in DEFAULT_UNDERCUT_BY_AGE.items():
        if lo <= opponent_tyre_age < hi:
            base_advantage = adv
            break
    else:
        if opponent_tyre_age >= 35:
            base_advantage = 1.32

    # Apply compound and circuit factors
    comp_factor = COMPOUND_FACTOR.get(new_compound.upper(), 1.0)
    return base_advantage * comp_factor * circuit_factor


def estimate_total_undercut_gain(
    opponent_tyre_age: int,
    new_compound: str = "MEDIUM",
    circuit_factor: float = 1.0,
    n_laps: int = UNDERCUT_WINDOW_LAPS,
) -> float:
    """Estimate total time gained over n_laps from the undercut.

    The advantage decays as the fresh tyres build temperature and the
    gain stabilises. We model this as a linear decay over the window.

    Args:
        opponent_tyre_age: How many laps the opponent has done on their tyres.
        new_compound: The compound you're switching to.
        circuit_factor: Circuit-specific multiplier.
        n_laps: Number of laps to accumulate the advantage over.

    Returns:
        Total time gained in seconds over the undercut window.
    """
    per_lap = estimate_undercut_advantage(opponent_tyre_age, new_compound, circuit_factor)

    # Linear decay: full advantage on lap 1, fading to ~30% by lap n_laps
    total = 0.0
    for i in range(n_laps):
        decay = 1.0 - 0.7 * (i / max(1, n_laps - 1))
        total += per_lap * decay
    return total


def compute_undercut_adjusted_pit_windows(
    base_windows: pd.DataFrame,
    current_tyre_age: int,
    pit_loss_seconds: float,
    new_compound_default: str = "MEDIUM",
    circuit_factor: float = 1.0,
    rival_tyre_age: int | None = None,
) -> pd.DataFrame:
    """Adjust pit window results to account for undercut dynamics.

    The key insight: the base calculator compares "your total time if you pit"
    vs "your total time if you don't". But it misses that the NO-STOP baseline
    should be PENALISED — if a rival pits and you don't, they gain time on you
    via the undercut. Staying out is not free.

    So instead of subtracting from pit strategies, we ADD an "undercut risk"
    penalty to the no-stop baseline: the time you'd lose if a rival pits and
    you don't. This makes pitting earlier look more attractive, matching real
    F1 behaviour.

    For each candidate pit lap N:
    - If YOU pit at lap N: you get the base strategy time (already computed)
    - If you DON'T pit by lap N: a rival who pits gains undercut_gain on you
    - So the "effective no-stop cost" at lap N includes the risk of being undercut

    Args:
        base_windows: Output of compute_pit_windows().
        current_tyre_age: Your tyre age at the decision point.
        pit_loss_seconds: Pit stop time loss.
        new_compound_default: Compound for undercut calculation.
        circuit_factor: Circuit-specific undercut strength.
        rival_tyre_age: Opponent's tyre age. If None, assumed same as yours.
    """
    adjusted = base_windows.copy()

    if rival_tyre_age is None:
        rival_tyre_age = current_tyre_age

    adjusted["undercut_gain"] = 0.0
    adjusted["net_pit_cost"] = pit_loss_seconds
    adjusted["undercut_adjusted_time"] = adjusted["expected_time"]
    adjusted["undercut_pct_recovered"] = 0.0

    # Get original no-stop time
    no_stop_mask = adjusted["pit_lap"] == 0
    no_stop_time = adjusted.loc[no_stop_mask, "expected_time"].values
    if len(no_stop_time) == 0:
        return adjusted
    no_stop_base = no_stop_time[0]

    for idx, row in adjusted.iterrows():
        if row["pit_lap"] == 0:
            continue

        pit_lap = int(row["pit_lap"])

        # Your tyre age at the point you'd pit
        your_age_at_pit = current_tyre_age + pit_lap

        # Parse compound from strategy string
        if "-> " in row["strategy"]:
            new_compound = row["strategy"].split("-> ")[-1]
        else:
            new_compound = new_compound_default

        # If a RIVAL pits at this lap, how much do they gain over you
        # while you stay out on old tyres?
        # The rival gets fresh tyres, you're on your_age_at_pit old tyres.
        undercut_risk = estimate_total_undercut_gain(
            opponent_tyre_age=your_age_at_pit,
            new_compound=new_compound,
            circuit_factor=circuit_factor,
        )

        # The pit strategy time stays the same (already accounts for fresh tyres)
        adjusted.loc[idx, "undercut_gain"] = undercut_risk
        adjusted.loc[idx, "net_pit_cost"] = pit_loss_seconds - undercut_risk
        adjusted.loc[idx, "undercut_adjusted_time"] = row["expected_time"]
        adjusted.loc[idx, "undercut_pct_recovered"] = (undercut_risk / pit_loss_seconds) * 100

    # Now recompute deltas: for each pit lap, the no-stop baseline is penalised
    # by the undercut risk at that lap
    for idx, row in adjusted.iterrows():
        if row["pit_lap"] == 0:
            adjusted.loc[idx, "undercut_adjusted_delta"] = 0.0
        else:
            # Effective delta = pit_strategy_time - (no_stop_time + undercut_risk)
            adjusted.loc[idx, "undercut_adjusted_delta"] = row["expected_time"] - (
                no_stop_base + row["undercut_gain"]
            )

    # Also update the no-stop row to show it has zero undercut adjustment
    adjusted.loc[no_stop_mask, "undercut_adjusted_time"] = no_stop_base
    adjusted.loc[no_stop_mask, "undercut_adjusted_delta"] = 0.0

    return adjusted


def compute_circuit_undercut_factors(df: pd.DataFrame) -> dict[str, float]:
    """Compute circuit-specific undercut strength factors from race data.

    Compares lap times of drivers on fresh tyres vs drivers on old tyres
    during the same race laps. Returns a multiplier relative to the
    global average (1.0 = average undercut strength).

    Args:
        df: Processed laps DataFrame.

    Returns:
        Dict mapping circuit_id -> undercut_factor.
    """
    records = []

    for _race_id, race_df in df.groupby("race_id"):
        race_df = race_df.sort_values(["driver_id", "lap_number"])
        circuit = race_df["circuit_id"].iloc[0]

        for driver_id, driver_df in race_df.groupby("driver_id"):
            driver_df = driver_df.sort_values("lap_number")

            # Detect stint transitions
            stint_changes = driver_df[driver_df["stint_number"].diff() > 0]

            for _, change in stint_changes.iterrows():
                pit_lap = int(change["lap_number"])

                # Fresh laps
                fresh = driver_df[
                    (driver_df["lap_number"] >= pit_lap)
                    & (driver_df["lap_number"] < pit_lap + 5)
                    & (driver_df["stint_number"] == change["stint_number"])
                ]
                if len(fresh) < 3:
                    continue

                # Compare to all drivers on old tyres at the same laps
                for other_id, other_df in race_df.groupby("driver_id"):
                    if other_id == driver_id:
                        continue
                    other_laps = other_df[
                        (other_df["lap_number"] >= pit_lap) & (other_df["lap_number"] < pit_lap + 5)
                    ]
                    if len(other_laps) < 3 or other_laps["tyre_life"].mean() < 10:
                        continue

                    merged = fresh[["lap_number", "lap_time_seconds"]].merge(
                        other_laps[["lap_number", "lap_time_seconds"]],
                        on="lap_number",
                        suffixes=("_fresh", "_old"),
                    )
                    if len(merged) < 3:
                        continue

                    advantage = (
                        merged["lap_time_seconds_old"] - merged["lap_time_seconds_fresh"]
                    ).mean()
                    records.append({"circuit": circuit, "advantage": advantage})

    if not records:
        return {}

    rdf = pd.DataFrame(records)
    global_mean = rdf["advantage"].mean()

    if global_mean <= 0:
        return {}

    factors = {}
    for circuit, group in rdf.groupby("circuit"):
        factors[circuit] = group["advantage"].mean() / global_mean

    logger.info(
        f"Circuit undercut factors: "
        f"min={min(factors.values()):.2f}, max={max(factors.values()):.2f}"
    )
    return factors
