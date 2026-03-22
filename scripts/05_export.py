#!/usr/bin/env python3
"""Export a trained model for Project B consumption."""

import argparse
import json
import logging
from pathlib import Path

from f1deg.config import load_config
from f1deg.export import export_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export a trained model")
    parser.add_argument("model", help="Model type to export")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    config = load_config(model_name=args.model)
    data_dir = Path(config["data_dir"])

    # Load model
    from scripts.train import get_model_class

    model_cls = get_model_class(args.model)
    model_dir = data_dir / "models" / args.model
    model = model_cls.load(model_dir)

    # Load metadata
    meta_path = data_dir / "processed" / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Export
    output_dir = Path(args.output) if args.output else data_dir / "models" / f"{args.model}_export"
    artifact_path = export_model(model, metadata, output_dir)
    logger.info(f"Model exported to {artifact_path}")


if __name__ == "__main__":
    main()
