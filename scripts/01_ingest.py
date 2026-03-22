#!/usr/bin/env python3
"""Ingest all race sessions from FastF1 and save as Parquet files."""

import logging

from f1deg.config import load_config
from f1deg.data.ingest import ingest_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    config = load_config()
    files = ingest_all(config)
    print(f"\nDone. Ingested {len(files)} race files.")


if __name__ == "__main__":
    main()
