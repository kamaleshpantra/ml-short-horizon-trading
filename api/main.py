from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

# Ensure src/ is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse
from api.services import PredictionService
from trading_ml.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger("api")

service: PredictionService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    logger.info("Initializing FastAPI prediction service...")
    service = PredictionService()
    yield


app = FastAPI(
    title="ML Short-Horizon Market Prediction API",
    description="Production microservice for short-horizon market direction forecasting",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service.get_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        return service.predict(request)
    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
