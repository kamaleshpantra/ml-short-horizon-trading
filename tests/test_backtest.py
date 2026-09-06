import numpy as np
import pandas as pd
import pytest

from trading_ml.backtesting.engine import BacktestEngine
from trading_ml.backtesting.metrics import calculate_trading_metrics
from trading_ml.backtesting.signal import SignalGenerator


def make_market_price_df(n_rows: int = 10) -> pd.DataFrame:
    mid = 100.0 + np.arange(n_rows) * 1.0  # 100, 101, 102, ...
    return pd.DataFrame({
        "mid_price": mid,
        "ask_price_1": mid + 0.1,
        "bid_price_1": mid - 0.1,
    })


def test_signal_generator():
    df = pd.DataFrame({"mid_price": [100.0, 101.0, 102.0]})
    # Probs: [P(DOWN), P(HOLD), P(UP)]
    probs = np.array([
        [0.1, 0.2, 0.7],  # UP (+1)
        [0.8, 0.1, 0.1],  # DOWN (-1)
        [0.3, 0.4, 0.3],  # FLAT (0)
    ])

    gen = SignalGenerator(buy_threshold=0.5, sell_threshold=0.5)
    signals = gen.generate_signals(df, probabilities=probs)

    assert list(signals) == [1.0, -1.0, 0.0]


def test_backtest_engine_execution_pricing():
    df = make_market_price_df(5)
    # Signals: Flat (0), Long (1), Long (1), Flat (0), Short (-1)
    signals = np.array([0.0, 1.0, 1.0, 0.0, -1.0])

    engine = BacktestEngine(
        fee_bps=10.0,  # 0.1%
        slippage_bps=0.0,
        use_bid_ask_spread=True,
        max_position=1.0,
        initial_capital=100000.0,
    )

    res = engine.run(df, signals)

    assert len(res.equity_curve) == 5
    assert len(res.trades) == 3  # Long open, Long close, Short open

    # First trade: Open Long at t=1. Ask price = 101.1
    t1 = res.trades[0]
    assert t1["side"] == "BUY"
    assert t1["exec_price"] == pytest.approx(101.1)
    assert t1["fee"] == pytest.approx(101.1 * 0.001)

    # Second trade: Close Long at t=3. Bid price = 103.0 - 0.1 = 102.9
    t2 = res.trades[1]
    assert t2["side"] == "SELL"
    assert t2["exec_price"] == pytest.approx(102.9)


def test_calculate_trading_metrics():
    equity = pd.Series([100000.0, 101000.0, 100500.0, 102000.0])
    trades = [
        {"pnl": 1000.0, "fee": 10.0, "slippage_cost": 5.0},
        {"pnl": -500.0, "fee": 10.0, "slippage_cost": 5.0},
        {"pnl": 1500.0, "fee": 10.0, "slippage_cost": 5.0},
    ]

    metrics = calculate_trading_metrics(equity, trades, initial_capital=100000.0)

    assert metrics["total_return"] == pytest.approx(0.02)
    assert metrics["final_capital"] == 102000.0
    assert metrics["total_trades"] == 3.0
    assert metrics["win_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["profit_factor"] == pytest.approx(2500.0 / 500.0)
    assert metrics["total_fees"] == 30.0


def test_empty_dataframe_backtest_error():
    df = pd.DataFrame()
    engine = BacktestEngine()

    with pytest.raises(ValueError, match="Cannot backtest empty DataFrame"):
        engine.run(df, np.array([]))
