#!/usr/bin/env python3
"""Backtest the pit window calculator against actual race pit stops.

For each race in the dataset, simulates being at lap N on stint 1 and asks
find_optimal_pit_lap() when to pit. Compares the recommendation against
the actual pit lap chosen by the race winner (and top-5 finishers).

Measures:
- % of races where recommendation is within ±2 laps of actual winner pit lap
- Mean absolute error between recommended and actual pit lap
- % of races where recommended compound matches actual
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from f1deg.config import load_config
from f1deg.models import get_model_class
from f1deg.strategy import find_optimal_pit_lap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def extract_actual_pit_stops(race_df: pd.DataFrame) -> pd.DataFrame:
    """Extract actual pit stop laps from race data by detecting stint transitions.

    Returns DataFrame with columns: driver_id, pit_lap, from_compound, to_compound,
    stint_number, finish_position.
    """
    records = []

    for driver_id, driver_df in race_df.groupby("driver_id"):
        driver_df = driver_df.sort_values("lap_number")
        # Finish position = last recorded position
        finish_pos = driver_df["position"].iloc[-1] if "position" in driver_df.columns else 99

        stints = (
            driver_df.groupby("stint_number")
            .agg(
                compound=("compound", "first"),
                first_lap=("lap_number", "min"),
                last_lap=("lap_number", "max"),
                n_laps=("lap_number", "count"),
            )
            .sort_index()
        )

        for i in range(len(stints) - 1):
            records.append(
                {
                    "driver_id": driver_id,
                    "pit_lap": int(stints.iloc[i]["last_lap"]),
                    "from_compound": stints.iloc[i]["compound"],
                    "to_compound": stints.iloc[i + 1]["compound"],
                    "stint_number": int(stints.index[i]),
                    "stint_laps": int(stints.iloc[i]["n_laps"]),
                    "finish_position": int(finish_pos),
                }
            )

    return pd.DataFrame(records)


def backtest_race(
    model,
    race_df: pd.DataFrame,
    race_id: str,
    config: dict,
) -> list[dict]:
    """Run the pit window calculator for a single race and compare to actuals.

    Simulates being at the start of the race (lap 1, stint 1) and asking
    the model for the optimal first pit stop.
    """
    results: list[dict] = []

    actual_stops = extract_actual_pit_stops(race_df)
    if actual_stops.empty:
        return results

    # Get race metadata
    circuit_id = race_df["circuit_id"].iloc[0]
    total_laps = int(race_df["lap_number"].max())

    # Get conditions from race data (median weather)
    conditions = {}
    for col in ["air_temp", "track_temp", "humidity", "wind_speed"]:
        if col in race_df.columns:
            conditions[col] = float(race_df[col].median())
    if "rainfall" in race_df.columns:
        conditions["rainfall"] = bool(race_df["rainfall"].any())

    # Focus on first pit stop of top-5 finishers
    first_stops = (
        actual_stops[actual_stops["stint_number"] == 1].sort_values("finish_position").head(5)
    )
    if first_stops.empty:
        return results

    # Winner's first stint compound
    winner_stop = first_stops.iloc[0]
    starting_compound = winner_stop["from_compound"]

    # Run optimizer
    try:
        result = find_optimal_pit_lap(
            model=model,
            circuit=circuit_id,
            total_laps=total_laps,
            current_compound=starting_compound,
            current_tyre_age=0,
            conditions=conditions,
        )
    except Exception as e:
        logger.warning(f"  {race_id}: optimizer failed: {e}")
        return results

    if result["optimal_pit_lap"] is None:
        logger.warning(f"  {race_id}: optimizer returned no pit recommendation")
        return results

    rec_lap = result["optimal_pit_lap"]
    rec_compound = result["optimal_compound"]

    # Compare against each top-5 finisher's first pit stop
    for _, stop in first_stops.iterrows():
        actual_lap = stop["pit_lap"]
        error = rec_lap - actual_lap
        within_2 = abs(error) <= 2
        within_3 = abs(error) <= 3
        within_5 = abs(error) <= 5
        compound_match = rec_compound == stop["to_compound"]

        results.append(
            {
                "race_id": race_id,
                "circuit_id": circuit_id,
                "total_laps": total_laps,
                "driver_id": stop["driver_id"],
                "finish_position": int(stop["finish_position"]),
                "starting_compound": starting_compound,
                "actual_pit_lap": actual_lap,
                "actual_to_compound": stop["to_compound"],
                "actual_stint_laps": int(stop["stint_laps"]),
                "recommended_pit_lap": rec_lap,
                "recommended_compound": rec_compound,
                "error_laps": error,
                "abs_error_laps": abs(error),
                "within_2_laps": within_2,
                "within_3_laps": within_3,
                "within_5_laps": within_5,
                "compound_match": compound_match,
                "time_saved_seconds": result["time_saved_seconds"],
                "crossover_lap": result["crossover_lap"],
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Backtest pit window calculator")
    parser.add_argument(
        "model",
        nargs="?",
        default="gbm",
        help="Model to use for predictions (default: gbm)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Seasons to backtest (default: all available)",
    )
    args = parser.parse_args()

    config = load_config()
    data_dir = Path(config["data_dir"])

    # Load data
    df = pd.read_parquet(data_dir / "processed" / "laps_clean.parquet")
    logger.info(f"Loaded {len(df)} laps")

    if args.seasons:
        df = df[df["season"].isin(args.seasons)]
        logger.info(f"Filtered to seasons {args.seasons}: {len(df)} laps")

    # Load model
    model_cls = get_model_class(args.model)
    model_dir = data_dir / "models" / args.model
    logger.info(f"Loading {args.model} model from {model_dir}")
    model = model_cls.load(model_dir)

    # Run backtest per race
    all_results = []
    race_ids = sorted(df["race_id"].unique())
    logger.info(f"Backtesting {len(race_ids)} races...")

    for i, race_id in enumerate(race_ids):
        race_df = df[df["race_id"] == race_id]

        # Skip races with no pit stops (red-flagged, abandoned, etc.)
        n_stints = race_df.groupby("driver_id")["stint_number"].nunique().max()
        if n_stints <= 1:
            logger.info(f"  [{i + 1}/{len(race_ids)}] {race_id}: no pit stops, skipping")
            continue

        results = backtest_race(model, race_df, race_id, config)
        all_results.extend(results)

        if (i + 1) % 10 == 0:
            logger.info(f"  [{i + 1}/{len(race_ids)}] processed")

    if not all_results:
        logger.error("No backtest results generated")
        return

    results_df = pd.DataFrame(all_results)

    # ── Headline metrics ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"PIT WINDOW BACKTEST — {args.model} model, {results_df['race_id'].nunique()} races")
    print(f"{'=' * 70}")

    # Winner-only metrics
    winners = results_df[results_df["finish_position"] == 1]
    top5 = results_df[results_df["finish_position"] <= 5]

    for label, subset in [("Winner", winners), ("Top 5", top5)]:
        n = len(subset)
        if n == 0:
            continue
        print(f"\n  {label} ({n} comparisons across {subset['race_id'].nunique()} races):")
        print(f"    Within ±2 laps:  {subset['within_2_laps'].mean():.1%}")
        print(f"    Within ±3 laps:  {subset['within_3_laps'].mean():.1%}")
        print(f"    Within ±5 laps:  {subset['within_5_laps'].mean():.1%}")
        print(f"    Mean abs error:  {subset['abs_error_laps'].mean():.1f} laps")
        print(f"    Median error:    {subset['error_laps'].median():+.1f} laps")
        print(f"    Compound match:  {subset['compound_match'].mean():.1%}")

    # ── Per-circuit breakdown ────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("PER-CIRCUIT ACCURACY (winner only)")
    print(f"{'─' * 70}")

    if len(winners) > 0:
        circuit_stats = (
            winners.groupby("circuit_id")
            .agg(
                n_races=("race_id", "nunique"),
                within_2=("within_2_laps", "mean"),
                mean_abs_err=("abs_error_laps", "mean"),
                median_err=("error_laps", "median"),
                compound_match=("compound_match", "mean"),
            )
            .sort_values("mean_abs_err")
        )

        print(f"\n{'circuit':<30} {'n':>3} {'±2':>6} {'MAE':>6} {'bias':>6} {'cmpd':>6}")
        print("-" * 60)
        for circuit, row in circuit_stats.iterrows():
            print(
                f"{circuit!s:<30} {int(row['n_races']):>3} "
                f"{row['within_2']:.0%}{'':<2} "
                f"{row['mean_abs_err']:>5.1f} "
                f"{row['median_err']:>+5.1f} "
                f"{row['compound_match']:.0%}"
            )

    # ── Bias analysis ────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("BIAS ANALYSIS")
    print(f"{'─' * 70}")

    if len(winners) > 0:
        mean_error = winners["error_laps"].mean()
        if mean_error > 1:
            print(f"  Model recommends pitting {mean_error:.1f} laps TOO LATE on average")
            print("  → Degradation model likely under-predicts late-stint deg")
        elif mean_error < -1:
            print(f"  Model recommends pitting {abs(mean_error):.1f} laps TOO EARLY on average")
            print("  → Degradation model likely over-predicts late-stint deg")
        else:
            print(f"  Mean error: {mean_error:+.1f} laps — well calibrated")

    # ── Save results ─────────────────────────────────────────────────────
    output_path = data_dir / "models" / f"backtest_{args.model}.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
