from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from trading_ml.data.parquet import write_parquet
from trading_ml.data.pipeline import build_processed_dataset
from trading_ml.utils.config import load_config
from trading_ml.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def load_raw_fi2010(path: str | Path) -> pd.DataFrame:
    """Load raw FI-2010 LOB matrix."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    df = pd.read_csv(path, header=None)
    logger.info("Loaded raw dataset: path=%s rows=%d cols=%d", path, len(df), len(df.columns))
    return df


def build_dataset_from_config(
    config_path: str = "configs/config.yaml", force: bool = False
) -> Path:
    """End-to-end reproducible build step for processed ML dataset."""
    configure_logging()
    config = load_config(config_path)

    symbol = config["data"]["symbol"]
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])

    target_cfg = config.get("target", {})
    horizon = target_cfg.get("horizon", 10)
    threshold = target_cfg.get("threshold", 0.0001)

    raw_path = raw_dir / "fi2010_sample.csv"
    output_path = processed_dir / "fi2010_processed.parquet"

    processed_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        logger.info("Processed dataset already exists at %s (use --force to overwrite)", output_path)
        return output_path

    logger.info("Starting processed dataset build for %s", symbol)
    raw = load_raw_fi2010(raw_path)

    processed = build_processed_dataset(
        raw,
        symbol=symbol,
        horizon=horizon,
        threshold=threshold,
        drop_na_targets=True,
    )

    write_parquet(processed, output_path)
    logger.info("Processed dataset successfully written to %s (rows=%d, cols=%d)", output_path, len(processed), len(processed.columns))

    if "target" in processed.columns:
        counts = processed["target"].value_counts().to_dict()
        logger.info("Target distribution: %s", counts)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed quantitative ML dataset.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if output exists")
    args = parser.parse_args()

    build_dataset_from_config(config_path=args.config, force=args.force)


if __name__ == "__main__":
    main()
