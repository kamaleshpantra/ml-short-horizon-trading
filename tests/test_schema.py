from datetime import datetime, timezone

import pytest

from trading_ml.data.schema import MarketEvent


def test_valid_market_event():

    event = MarketEvent(
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        bid_price_1=100.0,
        bid_size_1=10.0,
        ask_price_1=100.1,
        ask_size_1=12.0,
    )

    assert event.symbol == "BTCUSDT"
    assert event.bid_price_1 == 100.0


def test_negative_price_rejected():

    with pytest.raises(ValueError):

        MarketEvent(
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            bid_price_1=-100.0,
            bid_size_1=10.0,
            ask_price_1=100.1,
            ask_size_1=12.0,
        )


def test_negative_size_rejected():

    with pytest.raises(ValueError):

        MarketEvent(
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            bid_price_1=100.0,
            bid_size_1=-10.0,
            ask_price_1=100.1,
            ask_size_1=12.0,
        )