#!/usr/bin/env python3
"""Ingest sessions from FastF1 and save as Parquet files.

By default ingests Race sessions only (backward compatible).
Use --sessions to specify additional session types.

Examples:
    python scripts/01_ingest.py                       # Race only
    python scripts/01_ingest.py --sessions FP1,FP2,FP3,Q  # Practice + qualifying
    python scripts/01_ingest.py --sessions all         # All session types
"""

import argparse
import logging

from f1deg.config import load_config
from f1deg.data.ingest import ingest_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

ALL_SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "R"]


def main():
    parser = argparse.ArgumentParser(description="Ingest F1 session data from FastF1")
    parser.add_argument(
        "--sessions",
        type=str,
        default=None,
        help='Comma-separated session types (FP1,FP2,FP3,Q,R) or "all". '
        "Defaults to config value (R only).",
    )
    args = parser.parse_args()

    config = load_config()

    session_types = None
    if args.sessions:
        if args.sessions.lower() == "all":
            session_types = ALL_SESSION_TYPES
        else:
            session_types = [s.strip().upper() for s in args.sessions.split(",")]

    files = ingest_all(config, session_types=session_types)
    print(f"\nDone. Ingested {len(files)} session files.")


if __name__ == "__main__":
    main()
