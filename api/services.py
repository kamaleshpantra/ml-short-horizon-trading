from __future__ import annotations

from datetime import datetime, timezone
import logging
import sys
from pathlib import Path

# Ensure src/ is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from api.schemas import ModelInfoResponse, PredictionRequest, PredictionResponse
from trading_ml.data.parquet import read_parquet
from trading_ml.models.xgboost_model import XGBoostMarketPredictor
from trading_ml.utils.config import load_config

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "mid_price",
    "spread",
    "relative_spread",
    "obi_1",
    "obi_3",
    "obi_5",
    "obi_10",
    "microprice",
    "bid_depth_5",
    "ask_depth_5",
    "depth_ratio_5",
]

TARGET_LABEL_MAP = {1.0: "UP", 0.0: "HOLD", -1.0: "DOWN"}


class PredictionService:
    """Service encapsulating model inference lifecycle and online feature processing."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        self.model: XGBoostMarketPredictor | None = None
        self.version = "1.0.0"
        self._initialize_model()

    def _initialize_model(self) -> None:
        processed_path = Path("data/processed/fi2010_processed.parquet")
        if not processed_path.exists():
            from scripts.build_dataset import build_dataset_from_config
            logger.info("Building processed dataset for API service...")
            processed_path = build_dataset_from_config(self.config_path, force=True)

        df = read_parquet(processed_path)
        X = df[FEATURE_COLS]
        y = df["target"]

        logger.info("Fitting production prediction model on %d observations...", len(df))
        self.model = XGBoostMarketPredictor(n_estimators=50, max_depth=3, learning_rate=0.05)
        self.model.fit(X, y)
        logger.info("Production prediction service model initialized successfully.")

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        input_data = pd.DataFrame([req.model_dump()])
        X_mat = input_data[FEATURE_COLS]

        pred_val = self.model.predict(X_mat)[0]
        probs = self.model.predict_proba(X_mat)[0]  # [P(DOWN), P(HOLD), P(UP)]

        pred_label = TARGET_LABEL_MAP.get(pred_val, "HOLD")

        return PredictionResponse(
            prediction=pred_label,
            probability_down=float(probs[0]),
            probability_hold=float(probs[1]),
            probability_up=float(probs[2]),
            model_version=self.version,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_info(self) -> ModelInfoResponse:
        target_cfg = self.config.get("target", {})
        return ModelInfoResponse(
            model_type="XGBoostMarketPredictor",
            feature_list=FEATURE_COLS,
            horizon=target_cfg.get("horizon", 10),
            threshold=target_cfg.get("threshold", 0.0001),
            status="active",
        )
