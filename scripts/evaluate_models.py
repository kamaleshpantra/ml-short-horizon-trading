from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from trading_ml.data.parquet import read_parquet
from trading_ml.models.baseline import (
    LogisticRegressionBaseline,
    MajorityBaseline,
    OBIRuleBaseline,
)
from trading_ml.models.metrics import evaluate_classification
from trading_ml.models.xgboost_model import XGBoostMarketPredictor
from trading_ml.utils.config import load_config
from trading_ml.utils.logging import configure_logging
from trading_ml.validation.splits import temporal_train_val_test_split

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = [
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


def run_model_evaluation(
    data_path: str = "data/processed/fi2010_processed.parquet",
    config_path: str = "configs/config.yaml",
) -> pd.DataFrame:
    """Train and evaluate baseline and XGBoost models on time-series split."""
    configure_logging()
    config = load_config(config_path)

    path = Path(data_path)
    if not path.exists():
        from scripts.build_dataset import build_dataset_from_config
        logger.info("Processed dataset not found. Building dataset first...")
        path = build_dataset_from_config(config_path=config_path, force=True)

    df = read_parquet(path)
    logger.info("Loaded processed dataset for evaluation: rows=%d cols=%d", len(df), len(df.columns))

    split_cfg = config.get("split", {})
    purge_window = split_cfg.get("purge_window", 10)
    embargo_window = split_cfg.get("embargo_window", 0)

    train_df, val_df, test_df = temporal_train_val_test_split(
        df,
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
        purge_window=purge_window,
        embargo_window=embargo_window,
    )

    feature_cols = [col for col in DEFAULT_FEATURES if col in df.columns]
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    logger.info("Dataset splits: Train=%d Val=%d Test=%d", len(train_df), len(val_df), len(test_df))

    models = {
        "Majority": MajorityBaseline(),
        "OBI Rule": OBIRuleBaseline(obi_col="obi_1", threshold=0.2),
        "Logistic Regression": LogisticRegressionBaseline(C=1.0, max_iter=1000),
        "XGBoost": XGBoostMarketPredictor(n_estimators=50, max_depth=3, learning_rate=0.05),
    }

    results = []

    for name, model in models.items():
        logger.info("Fitting model: %s", name)
        model.fit(X_train, y_train)

        # Validation evaluation
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)
        val_metrics = evaluate_classification(y_val, val_preds, val_probs)

        # Test evaluation
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)
        test_metrics = evaluate_classification(y_test, test_preds, test_probs)

        results.append({
            "model": name,
            "val_acc": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_log_loss": val_metrics.get("log_loss", float("nan")),
            "test_acc": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_log_loss": test_metrics.get("log_loss", float("nan")),
        })

    summary_df = pd.DataFrame(results)
    logger.info("\n=== MODEL EVALUATION SUMMARY ===\n%s", summary_df.to_string(index=False))

    # Log XGBoost feature importances
    xgb_model = models["XGBoost"]
    if isinstance(xgb_model, XGBoostMarketPredictor):
        importances = xgb_model.get_feature_importances()
        logger.info("\n=== XGBoost Feature Importances ===\n%s", importances.to_string())

    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline ML models.")
    parser.add_argument("--data", default="data/processed/fi2010_processed.parquet", help="Path to processed dataset")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    run_model_evaluation(data_path=args.data, config_path=args.config)


if __name__ == "__main__":
    main()
