#!/usr/bin/env python3
"""Ingest race results (finishing status, retirements) from Jolpica API.

Saves one Parquet file per season to data/results/results_YYYY.parquet.
These are used by the feature pipeline to add retirement labels for
the anomaly prediction model.
"""

import logging
from pathlib import Path

from f1deg.config import load_config
from f1deg.data.results import get_season_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Maximum rounds to query per season (Jolpica returns empty for non-existent rounds)
MAX_ROUNDS_PER_SEASON = 30


def main():
    config = load_config()
    data_dir = Path(config["data_dir"])
    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    seasons = config.get("seasons", [2022, 2023, 2024, 2025])

    for year in seasons:
        output_path = results_dir / f"results_{year}.parquet"
        if output_path.exists():
            logger.info(f"Skipping {year} (already exists at {output_path})")
            continue

        logger.info(f"Fetching results for {year}...")

        # Count how many races we ingested for this season to set num_rounds
        raw_dir = data_dir / "raw"
        raw_files = list(raw_dir.glob(f"{year}_*.parquet"))
        num_rounds = len(raw_files) if raw_files else MAX_ROUNDS_PER_SEASON

        df = get_season_results(year, num_rounds)
        if df.empty:
            logger.warning(f"No results returned for {year}")
            continue

        df.to_parquet(output_path, index=False)
        retirements = df["did_retire"].sum()
        logger.info(
            f"Saved {len(df)} results for {year} to {output_path} ({retirements} retirements)"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
