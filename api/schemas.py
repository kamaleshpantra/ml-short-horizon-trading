from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input payload representing current order-book market state & microstructure features."""

    mid_price: float = Field(gt=0, description="Best bid/ask midpoint price")
    spread: float = Field(ge=0, description="Best ask - best bid spread")
    relative_spread: float = Field(ge=0, description="Spread normalized by mid_price")
    obi_1: float = Field(ge=-1.0, le=1.0, description="Level-1 order book imbalance")
    obi_3: float = Field(ge=-1.0, le=1.0, description="Level-3 order book imbalance")
    obi_5: float = Field(ge=-1.0, le=1.0, description="Level-5 order book imbalance")
    obi_10: float = Field(ge=-1.0, le=1.0, description="Level-10 order book imbalance")
    microprice: float = Field(gt=0, description="Volume-weighted level-1 microprice")
    bid_depth_5: float = Field(ge=0, description="Aggregate bid depth across 5 levels")
    ask_depth_5: float = Field(ge=0, description="Aggregate ask depth across 5 levels")
    depth_ratio_5: float = Field(ge=0, description="Bid depth / Ask depth ratio")


class PredictionResponse(BaseModel):
    """Output payload containing directional prediction and calibrated probabilities."""

    prediction: str = Field(description="Directional prediction: UP, HOLD, or DOWN")
    probability_up: float = Field(ge=0.0, le=1.0)
    probability_hold: float = Field(ge=0.0, le=1.0)
    probability_down: float = Field(ge=0.0, le=1.0)
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    version: str


class ModelInfoResponse(BaseModel):
    """Model information metadata response schema."""

    model_type: str
    feature_list: list[str]
    horizon: int
    threshold: float
    status: str
