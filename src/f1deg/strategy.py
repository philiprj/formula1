"""Pit window and crossover calculator for race strategy decisions.

Given a tire degradation model, computes the optimal pit lap by comparing
expected remaining race time for each strategy option at every lap.
"""

import logging

import pandas as pd

from f1deg.config import load_config
from f1deg.models.base import DegradationModel

logger = logging.getLogger(__name__)


def _remaining_race_time(
    model: DegradationModel,
    compound: str,
    circuit: str,
    current_lap: int,
    total_laps: int,
    tyre_age: int,
    fuel_kg: float,
    burn_rate: float,
    conditions: dict | None = None,
) -> tuple[float, float, float]:
    """Predict total remaining race time on current tires (no pit stop).

    Returns:
        (expected_time, lower_bound_time, upper_bound_time) in seconds.
    """
    remaining = total_laps - current_lap
    if remaining <= 0:
        return 0.0, 0.0, 0.0

    conds = dict(conditions or {})
    curve = model.predict_degradation_curve(
        compound=compound,
        circuit=circuit,
        n_laps=remaining + tyre_age,
        start_fuel_kg=fuel_kg + burn_rate * tyre_age,
        burn_rate=burn_rate,
        conditions=conds,
    )

    # Take only the laps from current tyre_age onward
    curve = curve[curve["lap_in_stint"] > tyre_age].reset_index(drop=True)
    if curve.empty:
        return float("inf"), float("inf"), float("inf")

    return (
        float(curve["predicted_lap_time"].sum()),
        float(curve["lower_bound"].sum()),
        float(curve["upper_bound"].sum()),
    )


def _pit_strategy_time(
    model: DegradationModel,
    current_compound: str,
    new_compound: str,
    circuit: str,
    pit_lap: int,
    total_laps: int,
    current_tyre_age: int,
    fuel_kg: float,
    burn_rate: float,
    pit_loss_seconds: float,
    conditions: dict | None = None,
) -> tuple[float, float, float]:
    """Predict total remaining time if pitting at pit_lap for new_compound.

    Time = laps_before_pit + pit_loss + laps_after_pit_on_fresh_tires.

    Returns:
        (expected_time, lower_bound_time, upper_bound_time) in seconds.
    """
    conds = dict(conditions or {})

    # Directly compute both phases (current tires until pit, then new tires after)

    # Phase 1: current tires until pit_lap
    phase1_laps = pit_lap
    if phase1_laps > 0:
        curve1 = model.predict_degradation_curve(
            compound=current_compound,
            circuit=circuit,
            n_laps=current_tyre_age + phase1_laps,
            start_fuel_kg=fuel_kg + burn_rate * current_tyre_age,
            burn_rate=burn_rate,
            conditions=conds,
        )
        curve1 = curve1[curve1["lap_in_stint"] > current_tyre_age].head(phase1_laps)
        time1 = float(curve1["predicted_lap_time"].sum())
        lower1 = float(curve1["lower_bound"].sum())
        upper1 = float(curve1["upper_bound"].sum())
    else:
        time1, lower1, upper1 = 0.0, 0.0, 0.0

    # Phase 2: fresh tires after pit
    phase2_laps = total_laps - pit_lap
    if phase2_laps > 0:
        fuel_at_pit = max(0.0, fuel_kg - burn_rate * phase1_laps)
        curve2 = model.predict_degradation_curve(
            compound=new_compound,
            circuit=circuit,
            n_laps=phase2_laps,
            start_fuel_kg=fuel_at_pit,
            burn_rate=burn_rate,
            conditions=conds,
        )
        time2 = float(curve2["predicted_lap_time"].sum())
        lower2 = float(curve2["lower_bound"].sum())
        upper2 = float(curve2["upper_bound"].sum())
    else:
        time2, lower2, upper2 = 0.0, 0.0, 0.0

    total = time1 + pit_loss_seconds + time2
    total_lower = lower1 + pit_loss_seconds + lower2
    total_upper = upper1 + pit_loss_seconds + upper2

    return total, total_lower, total_upper


def compute_pit_windows(
    model: DegradationModel,
    circuit: str,
    total_laps: int,
    current_compound: str,
    current_tyre_age: int = 0,
    fuel_kg: float | None = None,
    burn_rate: float | None = None,
    conditions: dict | None = None,
    candidate_compounds: list[str] | None = None,
    window_start: int | None = None,
    window_end: int | None = None,
) -> pd.DataFrame:
    """Compute expected remaining race time for each pit strategy at each lap.

    For each candidate pit lap within the window, compares:
    - No-stop: staying out on current tires to the finish
    - Pit to X: pitting at this lap for compound X

    Args:
        model: Trained degradation model.
        circuit: Circuit identifier (event name or key).
        total_laps: Total race laps.
        current_compound: Current tire compound.
        current_tyre_age: Laps already done on current tires.
        fuel_kg: Current fuel load. If None, estimated from lap number.
        burn_rate: Fuel burn rate. If None, loaded from config.
        conditions: Weather/driver conditions dict.
        candidate_compounds: Compounds to evaluate for pit stop. Defaults to dry compounds.
        window_start: First candidate pit lap (relative to remaining laps). Default: 1.
        window_end: Last candidate pit lap. Default: remaining_laps - 3.

    Returns:
        DataFrame with columns:
            pit_lap, strategy, expected_time, lower_bound, upper_bound, delta_vs_no_stop
    """
    config = load_config()
    fuel_config = config.get("fuel", {})

    if burn_rate is None:
        # Use circuit-specific rate if available, else global default
        circuit_fuel = config.get("fuel_by_circuit", {})
        from f1deg.data.features import _CIRCUIT_NAME_TO_KEY

        circuit_key = _CIRCUIT_NAME_TO_KEY.get(circuit, circuit)
        burn_rate = circuit_fuel.get(circuit_key, fuel_config.get("burn_rate_kg_per_lap", 1.5))

    if fuel_kg is None:
        start_mass = fuel_config.get("start_mass_kg", 110.0)
        laps_done = current_tyre_age  # approximate
        fuel_kg = max(0.0, start_mass - burn_rate * laps_done)

    pit_loss_map = config.get("pit_loss", {})
    from f1deg.data.features import _CIRCUIT_NAME_TO_KEY

    circuit_key = _CIRCUIT_NAME_TO_KEY.get(circuit, circuit)
    pit_loss = pit_loss_map.get(circuit_key, 23.0)

    if candidate_compounds is None:
        candidate_compounds = ["SOFT", "MEDIUM", "HARD"]

    remaining_laps = total_laps - current_tyre_age
    if window_start is None:
        window_start = 1
    if window_end is None:
        window_end = max(1, remaining_laps - 3)

    results = []

    # No-stop baseline at each candidate lap
    no_stop_time, no_stop_lower, no_stop_upper = _remaining_race_time(
        model=model,
        compound=current_compound,
        circuit=circuit,
        current_lap=total_laps - remaining_laps,
        total_laps=total_laps,
        tyre_age=current_tyre_age,
        fuel_kg=fuel_kg,
        burn_rate=burn_rate,
        conditions=conditions,
    )

    results.append(
        {
            "pit_lap": 0,
            "strategy": f"NO_STOP ({current_compound})",
            "expected_time": no_stop_time,
            "lower_bound": no_stop_lower,
            "upper_bound": no_stop_upper,
            "delta_vs_no_stop": 0.0,
        }
    )

    # Evaluate each pit lap / compound combination
    for pit_offset in range(window_start, window_end + 1):
        pit_lap = pit_offset  # laps from now until pit
        for compound in candidate_compounds:
            if compound == current_compound and current_tyre_age < 5:
                continue  # Skip same compound if tires are fresh

            strat_time, strat_lower, strat_upper = _pit_strategy_time(
                model=model,
                current_compound=current_compound,
                new_compound=compound,
                circuit=circuit,
                pit_lap=pit_lap,
                total_laps=remaining_laps,
                current_tyre_age=current_tyre_age,
                fuel_kg=fuel_kg,
                burn_rate=burn_rate,
                pit_loss_seconds=pit_loss,
                conditions=conditions,
            )

            results.append(
                {
                    "pit_lap": pit_lap,
                    "strategy": f"PIT_LAP_{pit_lap} -> {compound}",
                    "expected_time": strat_time,
                    "lower_bound": strat_lower,
                    "upper_bound": strat_upper,
                    "delta_vs_no_stop": strat_time - no_stop_time,
                }
            )

    df = pd.DataFrame(results)
    logger.info(
        f"Computed {len(df)} strategy options for {circuit}, "
        f"{remaining_laps} laps remaining on {current_compound} (age {current_tyre_age})"
    )
    return df


def find_optimal_pit_lap(
    model: DegradationModel,
    circuit: str,
    total_laps: int,
    current_compound: str,
    current_tyre_age: int = 0,
    fuel_kg: float | None = None,
    burn_rate: float | None = None,
    conditions: dict | None = None,
    candidate_compounds: list[str] | None = None,
    use_undercut: bool = True,
    rival_tyre_age: int | None = None,
    circuit_undercut_factor: float | None = None,
) -> dict:
    """Find the single best pit stop strategy.

    When use_undercut=True (default), adjusts the expected race time to
    account for the fresh-tyre advantage over rivals who stay out.
    This pulls the optimal pit lap earlier — matching real F1 behaviour
    where undercuts drive pit timing more than absolute degradation.

    Returns:
        Dict with keys: optimal_pit_lap, optimal_compound, time_saved_seconds,
        crossover_lap (first lap where pitting becomes beneficial),
        all_strategies (full DataFrame).
    """
    windows = compute_pit_windows(
        model=model,
        circuit=circuit,
        total_laps=total_laps,
        current_compound=current_compound,
        current_tyre_age=current_tyre_age,
        fuel_kg=fuel_kg,
        burn_rate=burn_rate,
        conditions=conditions,
        candidate_compounds=candidate_compounds,
    )

    # Apply empirical pit timing prior.
    #
    # The degradation model alone can't predict when to pit because:
    # 1. The "cliff" barely exists in race data (teams pit before it)
    # 2. Pit timing is driven by undercuts and game theory, not absolute deg
    #
    # Instead, we add an "overextension penalty" to strategies that pit
    # much later than the empirical median for that compound. This
    # captures the accumulated knowledge of real F1 strategists:
    # if everyone pits SOFT at lap 12, there's a reason.
    #
    # The penalty grows quadratically: small for ±2 laps, large for ±10.
    if use_undercut:
        from f1deg.strategy_undercut import EMPIRICAL_PIT_TIMING

        # Get typical stint length for this compound
        compound_timing = EMPIRICAL_PIT_TIMING.get(current_compound, {"median": 18, "std": 5})
        typical_stint = compound_timing["median"]
        stint_std = compound_timing["std"]

        for idx, row in windows.iterrows():
            if row["pit_lap"] == 0:
                continue

            pit_lap_offset = int(row["pit_lap"])
            # Total stint length at pit = current age + laps until pit
            total_stint = current_tyre_age + pit_lap_offset

            # How many standard deviations past the typical pit lap?
            z = (total_stint - typical_stint) / stint_std

            # Quadratic penalty for extending beyond typical stint length.
            # No penalty if pitting at or before the median.
            # 1 std late → small penalty. 2+ std → large penalty.
            if z > 0:
                # penalty in seconds: grows quadratically
                # At z=1 (~5-6 laps late): ~1.5s penalty
                # At z=2 (~10-12 laps late): ~6s penalty
                # At z=3 (~15-18 laps late): ~13.5s penalty
                penalty = 0.6 * z * z
                windows.loc[idx, "expected_time"] += penalty
                windows.loc[idx, "lower_bound"] += penalty
                windows.loc[idx, "upper_bound"] += penalty

        # Recompute deltas vs no-stop
        no_stop_time = windows.loc[windows["pit_lap"] == 0, "expected_time"].values
        if len(no_stop_time) > 0:
            windows["delta_vs_no_stop"] = windows["expected_time"] - no_stop_time[0]

    time_col = "expected_time"
    delta_col = "delta_vs_no_stop"

    pit_options = windows[windows["pit_lap"] > 0]
    if pit_options.empty:
        return {
            "optimal_pit_lap": None,
            "optimal_compound": None,
            "time_saved_seconds": 0.0,
            "crossover_lap": None,
            "all_strategies": windows,
        }

    best = pit_options.loc[pit_options[time_col].idxmin()]

    # Find crossover lap: first lap where any pit strategy beats no-stop
    crossover = None
    for lap in sorted(pit_options["pit_lap"].unique()):
        lap_options = pit_options[pit_options["pit_lap"] == lap]
        if (lap_options[delta_col] < 0).any():
            crossover = int(lap)
            break

    # Parse compound from strategy string
    compound = best["strategy"].split("-> ")[-1] if "-> " in best["strategy"] else None

    return {
        "optimal_pit_lap": int(best["pit_lap"]),
        "optimal_compound": compound,
        "time_saved_seconds": float(-best[delta_col]),
        "crossover_lap": crossover,
        "all_strategies": windows,
    }


def compute_sc_adjusted_pit_windows(
    model: DegradationModel,
    circuit: str,
    total_laps: int,
    current_compound: str,
    current_tyre_age: int = 0,
    sc_probability_per_lap: float | None = None,
    sc_pit_saving_seconds: float = 18.0,
    fuel_kg: float | None = None,
    burn_rate: float | None = None,
    conditions: dict | None = None,
    candidate_compounds: list[str] | None = None,
    sc_rates: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Compute pit windows adjusted for safety car probability.

    For each candidate pit lap, adjusts the expected time by the probability
    of a safety car occurring (which would make the pit stop cheaper).

    The adjustment: at each future lap, there's a sc_probability_per_lap chance
    of SC, which saves ~sc_pit_saving_seconds on pit loss. The expected value
    of waiting is discounted by the cumulative probability of SC.

    Args:
        sc_probability_per_lap: SC probability per lap. If None, uses circuit-specific
            rate from sc_rates (or falls back to 4%).
        sc_pit_saving_seconds: Time saved by pitting under SC vs green (default 18s).
        sc_rates: Pre-computed SC rates DataFrame from strategy_sc.compute_sc_rates_per_circuit().
    """
    # Use circuit-specific SC rate if not explicitly provided
    if sc_probability_per_lap is None:
        try:
            from f1deg.strategy_sc import get_sc_probability

            sc_probability_per_lap = get_sc_probability(circuit, sc_rates=sc_rates)
            logger.info(
                f"Using circuit-specific SC rate for {circuit}: {sc_probability_per_lap:.3f}"
            )
        except Exception:
            sc_probability_per_lap = 0.04

    base_windows = compute_pit_windows(
        model=model,
        circuit=circuit,
        total_laps=total_laps,
        current_compound=current_compound,
        current_tyre_age=current_tyre_age,
        fuel_kg=fuel_kg,
        burn_rate=burn_rate,
        conditions=conditions,
        candidate_compounds=candidate_compounds,
    )

    # For each pit option, compute the expected SC benefit of waiting
    adjusted = base_windows.copy()
    adjusted["sc_expected_saving"] = 0.0
    adjusted["sc_adjusted_time"] = adjusted["expected_time"]

    for idx, row in adjusted.iterrows():
        if row["pit_lap"] == 0:
            continue  # no-stop doesn't benefit from SC timing

        pit_lap = int(row["pit_lap"])
        # Probability that SC occurs between now and pit_lap
        p_no_sc_before_pit = (1 - sc_probability_per_lap) ** pit_lap
        # Expected savings from SC-timed pit stop
        # If SC occurs at any lap before our planned pit, we can pit under SC
        p_sc_before_pit = 1 - p_no_sc_before_pit
        sc_saving = p_sc_before_pit * sc_pit_saving_seconds

        adjusted.loc[idx, "sc_expected_saving"] = sc_saving
        adjusted.loc[idx, "sc_adjusted_time"] = row["expected_time"] - sc_saving

    # Recompute deltas against no-stop using adjusted times
    no_stop_time = adjusted.loc[adjusted["pit_lap"] == 0, "sc_adjusted_time"].values
    if len(no_stop_time) > 0:
        adjusted["sc_adjusted_delta"] = adjusted["sc_adjusted_time"] - no_stop_time[0]
    else:
        adjusted["sc_adjusted_delta"] = 0.0

    return adjusted
