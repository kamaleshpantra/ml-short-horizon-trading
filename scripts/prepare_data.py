from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from trading_ml.data.normalization import normalize_fi2010
from trading_ml.data.parquet import write_parquet
from trading_ml.data.validation import validate_market_data
from trading_ml.utils.config import load_config
from trading_ml.utils.logging import configure_logging


logger = logging.getLogger(__name__)


def load_raw_fi2010(path: str | Path) -> pd.DataFrame:
    """Load a raw FI-2010 matrix."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        header=None,
    )

    logger.info(
        "Loaded raw dataset: rows=%d columns=%d",
        len(df),
        len(df.columns),
    )

    return df


def main() -> None:
    configure_logging()

    config = load_config("configs/config.yaml")

    symbol = config["data"]["symbol"]
    raw_dir = Path(config["data"]["raw_dir"])
    interim_dir = Path(config["data"]["interim_dir"])

    raw_path = raw_dir / "fi2010_sample.csv"
    output_path = (
        interim_dir / "fi2010_normalized.parquet"
    )

    logger.info("Starting data preparation")
    logger.info("Symbol: %s", symbol)

    raw = load_raw_fi2010(raw_path)

    normalized = normalize_fi2010(
        raw,
        symbol=symbol,
    )

    logger.info(
        "Normalized dataset: rows=%d columns=%d",
        len(normalized),
        len(normalized.columns),
    )

    # Our existing validator expects timestamp.
    # FI-2010 does not provide exchange timestamps in this
    # raw matrix, so validation of the historical representation
    # will be added separately.
    validate_market_data(
        _build_validation_frame(normalized)
    )

    write_parquet(
        normalized,
        output_path,
    )

    logger.info(
        "Normalized dataset written to %s",
        output_path,
    )


def _build_validation_frame(
    normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the minimal representation required by the
    generic market-data validator.
    """

    validation = pd.DataFrame(
        {
            "timestamp": normalized["event_id"],
            "bid_price_1": normalized["bid_price_1"],
            "bid_size_1": normalized["bid_size_1"],
            "ask_price_1": normalized["ask_price_1"],
            "ask_size_1": normalized["ask_size_1"],
        }
    )

    return validation


if __name__ == "__main__":
    main()