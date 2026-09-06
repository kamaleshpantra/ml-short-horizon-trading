from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from trading_ml.backtesting.engine import BacktestEngine
from trading_ml.backtesting.signal import SignalGenerator
from trading_ml.data.parquet import read_parquet
from trading_ml.models.baseline import (
    LogisticRegressionBaseline,
    MajorityBaseline,
    OBIRuleBaseline,
)
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


def run_backtest_pipeline(
    data_path: str = "data/processed/fi2010_processed.parquet",
    config_path: str = "configs/config.yaml",
    fee_bps: float = 5.0,
    slippage_bps: float = 1.0,
    buy_threshold: float = 0.50,
) -> pd.DataFrame:
    """Execute end-to-end model backtest and financial performance analysis."""
    configure_logging()
    config = load_config(config_path)

    path = Path(data_path)
    if not path.exists():
        from scripts.build_dataset import build_dataset_from_config
        logger.info("Processed dataset not found. Building dataset first...")
        path = build_dataset_from_config(config_path=config_path, force=True)

    df = read_parquet(path)
    logger.info("Loaded dataset for backtesting: rows=%d cols=%d", len(df), len(df.columns))

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
    X_test = test_df[feature_cols]

    logger.info("Out-of-sample Test Period: rows=%d events", len(test_df))

    models = {
        "Majority": MajorityBaseline(),
        "OBI Rule": OBIRuleBaseline(obi_col="obi_1", threshold=0.2),
        "Logistic Regression": LogisticRegressionBaseline(C=1.0, max_iter=1000),
        "XGBoost": XGBoostMarketPredictor(n_estimators=50, max_depth=3, learning_rate=0.05),
    }

    signal_gen = SignalGenerator(buy_threshold=buy_threshold, sell_threshold=buy_threshold)
    engine = BacktestEngine(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        use_bid_ask_spread=True,
        max_position=1.0,
        initial_capital=100000.0,
    )

    backtest_results = []

    for name, model in models.items():
        logger.info("Fitting and backtesting model: %s", name)
        model.fit(X_train, y_train)

        test_probs = model.predict_proba(X_test)
        signals = signal_gen.generate_signals(test_df, probabilities=test_probs)

        result = engine.run(test_df, signals)
        m = result.metrics

        backtest_results.append({
            "Model": name,
            "Total Return (%)": round(m.get("total_return", 0.0) * 100, 2),
            "Sharpe Ratio": round(m.get("sharpe_ratio", 0.0), 3),
            "Sortino Ratio": round(m.get("sortino_ratio", 0.0), 3),
            "Max Drawdown (%)": round(m.get("max_drawdown", 0.0) * 100, 2),
            "Win Rate (%)": round(m.get("win_rate", 0.0) * 100, 1),
            "Profit Factor": round(m.get("profit_factor", 0.0), 2),
            "Trades": int(m.get("total_trades", 0)),
            "Total Fees ($)": round(m.get("total_fees", 0.0), 2),
            "Slippage Cost ($)": round(m.get("total_slippage_cost", 0.0), 2),
        })

    summary_df = pd.DataFrame(backtest_results)
    logger.info("\n=== OUT-OF-SAMPLE BACKTEST PERFORMANCE (Fee=%g bps, Slippage=%g bps) ===\n%s", fee_bps, slippage_bps, summary_df.to_string(index=False))

    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run realistic backtest on ML models.")
    parser.add_argument("--data", default="data/processed/fi2010_processed.parquet", help="Path to processed dataset")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--fee", type=float, default=5.0, help="Transaction fee in bps")
    parser.add_argument("--slippage", type=float, default=1.0, help="Slippage in bps")
    args = parser.parse_args()

    run_backtest_pipeline(data_path=args.data, config_path=args.config, fee_bps=args.fee, slippage_bps=args.slippage)


if __name__ == "__main__":
    main()
