from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MarketEvent(BaseModel):
    """Canonical representation of a real-time market-data event."""

    timestamp: datetime
    symbol: str

    bid_price_1: float = Field(gt=0)
    bid_size_1: float = Field(ge=0)

    ask_price_1: float = Field(gt=0)
    ask_size_1: float = Field(ge=0)

    bid_price_2: float | None = Field(default=None, gt=0)
    bid_size_2: float | None = Field(default=None, ge=0)

    ask_price_2: float | None = Field(default=None, gt=0)
    ask_size_2: float | None = Field(default=None, ge=0)

    bid_price_3: float | None = Field(default=None, gt=0)
    bid_size_3: float | None = Field(default=None, ge=0)

    ask_price_3: float | None = Field(default=None, gt=0)
    ask_size_3: float | None = Field(default=None, ge=0)

    bid_price_4: float | None = Field(default=None, gt=0)
    bid_size_4: float | None = Field(default=None, ge=0)

    ask_price_4: float | None = Field(default=None, gt=0)
    ask_size_4: float | None = Field(default=None, ge=0)

    bid_price_5: float | None = Field(default=None, gt=0)
    bid_size_5: float | None = Field(default=None, ge=0)

    ask_price_5: float | None = Field(default=None, gt=0)
    ask_size_5: float | None = Field(default=None, ge=0)

    trade_price: float | None = Field(default=None, gt=0)
    trade_size: float | None = Field(default=None, ge=0)

    trade_side: str | None = None


class HistoricalLOBRecord(BaseModel):
    """Canonical representation of event-indexed historical LOB benchmark records."""

    event_id: int = Field(ge=0)
    symbol: str

    bid_price_1: float = Field(gt=0)
    bid_size_1: float = Field(ge=0)

    ask_price_1: float = Field(gt=0)
    ask_size_1: float = Field(ge=0)