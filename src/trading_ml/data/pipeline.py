from __future__ import annotations

import logging

import pandas as pd

from trading_ml.data.normalization import normalize_fi2010
from trading_ml.data.validation import validate_market_data
from trading_ml.features.microstructure import build_microstructure_features
from trading_ml.targets.returns import build_targets

logger = logging.getLogger(__name__)


def build_processed_dataset(
    raw_df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    horizon: int = 10,
    threshold: float = 0.0001,
    drop_na_targets: bool = True,
) -> pd.DataFrame:
    """
    Transform raw FI-2010 LOB matrix into a feature-rich, labeled ML dataset.

    Pipeline steps:
    1. Normalize raw LOB format to standard column names
    2. Validate price, size, schema, and order constraints
    3. Generate order-book microstructure features (mid-price, spread, OBI, microprice, depth)
    4. Compute leakage-free future returns and target direction labels
    5. Optionally drop rows with unknown future targets (the final H observations)

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw input LOB matrix.
    symbol : str, default="BTCUSDT"
        Asset ticker symbol.
    horizon : int, default=10
        Future return horizon H in events.
    threshold : float, default=0.0001
        Direction classification threshold.
    drop_na_targets : bool, default=True
        Whether to drop rows where target is NaN (e.g. final H observations).

    Returns
    -------
    pd.DataFrame
        Processed dataset ready for ML modeling.
    """
    logger.info("Normalizing raw LOB data: rows=%d", len(raw_df))
    normalized = normalize_fi2010(raw_df, symbol=symbol)

    logger.info("Validating normalized market data")
    validation_df = pd.DataFrame(
        {
            "timestamp": normalized["event_id"],
            "bid_price_1": normalized["bid_price_1"],
            "bid_size_1": normalized["bid_size_1"],
            "ask_price_1": normalized["ask_price_1"],
            "ask_size_1": normalized["ask_size_1"],
        }
    )
    validate_market_data(validation_df)

    logger.info("Building microstructure features")
    features = build_microstructure_features(normalized)

    logger.info(
        "Building target labels: horizon=%d threshold=%.6f", horizon, threshold
    )
    processed = build_targets(
        features,
        horizon=horizon,
        threshold=threshold,
    )

    if drop_na_targets:
        before_len = len(processed)
        processed = processed.dropna(subset=["target"]).copy()
        logger.info(
            "Dropped NaN targets: %d -> %d rows", before_len, len(processed)
        )

    return processed
