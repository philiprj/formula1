#!/usr/bin/env python3
"""Compare all trained models side-by-side.

Loads each trained model, predicts on the same holdout races, and produces
a residual breakdown by compound, circuit, stint phase, and SC proximity.
"""

import argparse
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from f1deg.config import load_config
from f1deg.eval.diagnostics import (
    compare_model_residuals,
    identify_bayesian_improvement_areas,
    residual_analysis,
)
from f1deg.eval.metrics import compute_all_metrics
from f1deg.models import MODEL_REGISTRY, get_model_class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compare all trained models")
    parser.add_argument(
        "--holdout-races",
        type=int,
        default=5,
        help="Number of most recent races to hold out (default: 5)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to compare (default: all trained models found on disk)",
    )
    args = parser.parse_args()

    config = load_config()
    data_dir = Path(config["data_dir"])

    # Load data
    df = pd.read_parquet(data_dir / "processed" / "laps_clean.parquet")
    race_ids = sorted(df["race_id"].unique())
    logger.info(f"Loaded {len(df)} laps, {len(race_ids)} races")

    # Holdout split
    holdout_races = min(args.holdout_races, max(1, len(race_ids) // 5))
    test_races = race_ids[-holdout_races:]
    test_df = df[df["race_id"].isin(test_races)].copy()
    y_true = test_df["lap_time_seconds"].values
    logger.info(f"Holdout: {len(test_races)} races, {len(test_df)} laps")

    # Discover which models are trained
    model_names = args.models or list(MODEL_REGISTRY.keys())
    available = []
    for name in model_names:
        model_dir = data_dir / "models" / name
        if model_dir.exists():
            available.append(name)
        else:
            logger.warning(f"Skipping {name}: no trained model at {model_dir}")

    if len(available) < 2:
        logger.error("Need at least 2 trained models to compare")
        raise SystemExit(1)

    # Load models and predict
    results = {}  # model_name -> (y_true, y_pred)
    metrics = {}  # model_name -> metrics dict
    for name in available:
        model_cls = get_model_class(name)
        model_dir = data_dir / "models" / name
        logger.info(f"Loading {name} from {model_dir}")
        try:
            t0 = time.time()
            model = model_cls.load(model_dir)
            logger.info(f"  {name} loaded in {time.time() - t0:.1f}s")

            t0 = time.time()
            y_pred = model.predict(test_df)
            logger.info(f"  {name} predict() in {time.time() - t0:.1f}s")

            t0 = time.time()
            lower, upper = model.predict_interval(test_df)
            logger.info(f"  {name} predict_interval() in {time.time() - t0:.1f}s")

            results[name] = (y_true, y_pred)
            metrics[name] = compute_all_metrics(y_true, y_pred, lower, upper, test_df)
        except Exception as e:
            logger.warning(f"  {name} failed: {e} — skipping")
            available = [m for m in available if m != name]

    # ── Headline comparison ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"MODEL COMPARISON — {len(test_races)} holdout races, {len(test_df)} laps")
    print(f"{'=' * 70}")

    header_keys = [
        "mae",
        "rmse",
        "degradation_trend_accuracy",
        "mae_stint_early",
        "mae_stint_mid",
        "mae_stint_late",
        "mae_post_sc",
        "pi_coverage_95",
        "pi_width_mean",
    ]

    # Print header
    col_w = 12
    print(f"\n{'metric':<30}", end="")
    for name in available:
        print(f"{name:>{col_w}}", end="")
    print()
    print("-" * (30 + col_w * len(available)))

    for key in header_keys:
        print(f"{key:<30}", end="")
        values = [metrics[name].get(key, float("nan")) for name in available]
        best = (
            min(v for v in values if not np.isnan(v))
            if any(not np.isnan(v) for v in values)
            else None
        )
        for v in values:
            if np.isnan(v):
                cell = "N/A"
            elif key == "degradation_trend_accuracy":
                # Higher is better for trend accuracy
                cell = f"{v:.4f}"
                if best is not None and v == max(vv for vv in values if not np.isnan(vv)):
                    cell += " ★"
            elif key == "pi_coverage_95":
                # Closest to 0.95 is best
                cell = f"{v:.4f}"
                if best is not None and abs(v - 0.95) == min(
                    abs(vv - 0.95) for vv in values if not np.isnan(vv)
                ):
                    cell += " ★"
            else:
                cell = f"{v:.4f}"
                if best is not None and v == best:
                    cell += " ★"
            print(f"{cell:>{col_w}}", end="")
        print()

    # ── Residual breakdown ───────────────────────────────────────────────
    combined = compare_model_residuals(results, test_df)

    for dimension in ["compound", "circuit_id", "stint_phase", "sc_proximity"]:
        dim_data = combined[combined["dimension"] == dimension]
        if dim_data.empty:
            continue

        print(f"\n{'─' * 70}")
        print(f"BREAKDOWN BY: {dimension.upper()}")
        print(f"{'─' * 70}")

        dim_data.pivot_table(
            index="group",
            columns="model",
            values=["mae", "bias"],
            aggfunc="first",
        )

        # Print MAE sub-table
        print(f"\n{'group':<25}", end="")
        for name in available:
            print(f"{'MAE ' + name:>{col_w + 4}}", end="")
        print(f"{'best':>{col_w}}")

        groups = sorted(dim_data["group"].unique())
        for group in groups:
            print(f"{group!s:<25}", end="")
            row_values = {}
            for name in available:
                match = dim_data[(dim_data["group"] == group) & (dim_data["model"] == name)]
                val = match["mae"].values[0] if len(match) > 0 else float("nan")
                row_values[name] = val
                print(f"{val:>{col_w + 4}.4f}", end="")

            valid = {k: v for k, v in row_values.items() if not np.isnan(v)}
            if valid:
                best_model = min(valid, key=valid.get)
                print(f"{best_model:>{col_w}}", end="")
            print()

        # Print bias sub-table
        print(f"\n{'group':<25}", end="")
        for name in available:
            print(f"{'bias ' + name:>{col_w + 4}}", end="")
        print()

        for group in groups:
            print(f"{group!s:<25}", end="")
            for name in available:
                match = dim_data[(dim_data["group"] == group) & (dim_data["model"] == name)]
                val = match["bias"].values[0] if len(match) > 0 else float("nan")
                sign = "+" if val > 0 else ""
                print(f"{sign}{val:>{col_w + 3}.4f}", end="")
            print()

    # ── Bayesian improvement suggestions ─────────────────────────────────
    if "bayesian" in available and "gbm" in available:
        print(f"\n{'=' * 70}")
        print("BAYESIAN vs GBM — IMPROVEMENT SUGGESTIONS")
        print(f"{'=' * 70}")

        bay_residuals = residual_analysis(*results["bayesian"], test_df, "bayesian")
        gbm_residuals = residual_analysis(*results["gbm"], test_df, "gbm")
        suggestions = identify_bayesian_improvement_areas(bay_residuals, gbm_residuals)

        if suggestions:
            for i, s in enumerate(suggestions, 1):
                print(f"\n  {i}. {s['dimension']}/{s['group']}")
                print(
                    f"     Bayesian MAE: {s['bayesian_mae']:.3f}s  |  GBM MAE: {s['gbm_mae']:.3f}s  |  Gap: {s['gap']:.3f}s"
                )
                print(f"     Bias: {s['bayesian_bias']:+.3f}s  ({s['n_laps']} laps)")
                print(f"     → {s['suggestion']}")
        else:
            print("  No significant gaps found — Bayesian model competitive with GBM.")

    # ── Save full comparison ─────────────────────────────────────────────
    output_path = data_dir / "models" / "model_comparison.csv"
    combined.to_csv(output_path, index=False)
    logger.info(f"Full comparison saved to {output_path}")


if __name__ == "__main__":
    main()
