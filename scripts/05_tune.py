#!/usr/bin/env python3
"""Hyperparameter tuning for degradation models.

Usage:
    python scripts/05_tune.py gbm                          # XGBoost, 150 trials
    python scripts/05_tune.py gbm --backend lightgbm       # LightGBM tuning
    python scripts/05_tune.py gbm --n-trials 50            # Quick 50-trial search
    python scripts/05_tune.py gbm --max-folds 5            # Fewer CV folds per trial
    python scripts/05_tune.py gbm --apply                  # Tune + write best params to yaml
"""

import argparse
import json
import logging
from pathlib import Path
import re

import pandas as pd

from f1deg.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Project root — conf/ lives here
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _apply_params_to_yaml(yaml_path: Path, backend: str, best_params: dict) -> None:
    """Update the backend section of a YAML config with tuned hyperparameters.

    Uses line-level text replacement to preserve comments and formatting.
    Only updates keys that exist in best_params within the target backend section.
    """
    text = yaml_path.read_text()
    lines = text.splitlines(keepends=True)

    # Find the backend section (e.g. "xgboost:" or "lightgbm:")
    section_pattern = re.compile(rf"^{re.escape(backend)}\s*:")
    in_section = False
    indent = ""
    updated_keys = set()
    new_lines = []

    for line in lines:
        stripped = line.lstrip()
        current_indent = line[: len(line) - len(stripped)]

        if section_pattern.match(stripped):
            # Entering the target section
            in_section = True
            indent = ""  # will be set by the first key line
            new_lines.append(line)
            continue

        if in_section:
            # Detect end of section: a non-empty, non-comment line with same or less indent
            if (
                stripped
                and not stripped.startswith("#")
                and indent
                and len(current_indent) <= len(indent)
                and ":" in stripped
                and (len(current_indent) == 0 or (indent and len(current_indent) < len(indent)))
            ):
                in_section = False
                new_lines.append(line)
                continue

            if stripped and not stripped.startswith("#"):
                # First key sets the expected indent
                if not indent and ":" in stripped:
                    indent = current_indent

                # Try to match a key: value line
                key_match = re.match(r"(\w[\w_]*):\s*(.*)", stripped)
                if key_match and indent and len(current_indent) == len(indent):
                    key = key_match.group(1)
                    if key in best_params:
                        value = best_params[key]
                        formatted = f"{value:.8g}" if isinstance(value, float) else str(value)

                        # Preserve any inline comment
                        old_value_and_comment = key_match.group(2)
                        comment_match = re.search(r"\s*(#.*)$", old_value_and_comment)
                        comment = comment_match.group(1) if comment_match else ""
                        if comment:
                            new_line = f"{indent}{key}: {formatted}  {comment}\n"
                        else:
                            new_line = f"{indent}{key}: {formatted}\n"

                        new_lines.append(new_line)
                        updated_keys.add(key)
                        continue

        new_lines.append(line)

    # Write back
    yaml_path.write_text("".join(new_lines))

    # Report
    not_updated = set(best_params.keys()) - updated_keys
    if updated_keys:
        logger.info(f"Updated {len(updated_keys)} params in {yaml_path}: {sorted(updated_keys)}")
    if not_updated:
        logger.warning(
            f"Could not find keys to update in {backend} section: {sorted(not_updated)}. "
            f"You may need to add them manually."
        )


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for degradation models")
    parser.add_argument(
        "model",
        choices=["gbm"],
        help="Model type to tune (currently only 'gbm' is supported)",
    )
    parser.add_argument(
        "--backend",
        choices=["xgboost", "lightgbm"],
        default=None,
        help="Override backend (default: use config value)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials (default: from config, typically 150)",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Max LORO CV folds per trial (default: from config, typically 10)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write best hyperparameters directly into the model's YAML config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save tuning results JSON (default: data/models/<model>/tuning_results.json)",
    )
    args = parser.parse_args()

    config = load_config(model_name=args.model)
    data_dir = Path(config["data_dir"])

    # Override backend if specified
    if args.backend:
        config.setdefault("model", {})["backend"] = args.backend

    # Load processed data
    data_path = data_dir / "processed" / "laps_clean.parquet"
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} laps, {df['race_id'].nunique()} races")

    # Resolve tuning parameters
    tuning_config = config.get("model", {}).get("tuning", {})
    n_trials = args.n_trials or tuning_config.get("n_trials", 150)
    max_folds = args.max_folds or tuning_config.get("max_folds", 10)

    backend = config.get("model", {}).get("backend", "xgboost")
    logger.info(f"Starting {args.model} ({backend}) tuning: {n_trials} trials, {max_folds} folds")

    # Run tuning
    if args.model == "gbm":
        from f1deg.eval.tuning import tune_gbm

        results = tune_gbm(df, config, n_trials=n_trials, max_folds=max_folds)
    else:
        raise ValueError(f"Tuning not implemented for model: {args.model}")

    # Print results
    print(f"\n=== Tuning Results ({backend}) ===")
    print(f"  Best MAE: {results['best_mae']:.4f}s")
    print(f"  Trials: {results['n_trials']} ({results.get('n_pruned', 0)} pruned)")
    print("  Best params:")
    for key, value in sorted(results["best_params"].items()):
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

    print("\n  Top 5 trials:")
    for t in results["trials_summary"]:
        print(f"    Trial {t['number']}: MAE={t['mae']:.4f}s")

    # Save results JSON
    output_path = args.output or str(data_dir / "models" / args.model / "tuning_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    # Apply to YAML if requested
    if args.apply:
        yaml_path = _PROJECT_ROOT / "conf" / "models" / f"{args.model}.yaml"
        if not yaml_path.exists():
            logger.error(f"Config file not found: {yaml_path}")
            return

        _apply_params_to_yaml(yaml_path, backend, results["best_params"])
        print(f"\n✓ Best params written to {yaml_path} [{backend}] section")
        print(f"  Run 'make train MODEL={args.model}' to train with tuned params")
    else:
        print("\nTo apply these params, either:")
        print(f"  1. Re-run with --apply:  python scripts/05_tune.py {args.model} --apply")
        print(f"  2. Manually update conf/models/{args.model}.yaml [{backend}] section")


if __name__ == "__main__":
    main()
