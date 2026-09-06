from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.backtesting.metrics import calculate_trading_metrics


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[dict[str, Any]]
    metrics: dict[str, float]
    positions: pd.Series


class BacktestEngine:
    """
    Event-driven Realistic Execution Backtesting Engine for short-horizon LOB trading.

    Key Features:
    - Microstructure execution: Buys at Ask_1, Sells at Bid_1
    - Configurable transaction fees (fee_bps)
    - Configurable slippage (slippage_bps)
    - Strict position limits (max_position)
    """

    def __init__(
        self,
        fee_bps: float = 5.0,
        slippage_bps: float = 1.0,
        use_bid_ask_spread: bool = True,
        max_position: float = 1.0,
        initial_capital: float = 100000.0,
    ) -> None:
        self.fee_bps = fee_bps
        self.fee_rate = fee_bps / 10000.0

        self.slippage_bps = slippage_bps
        self.slippage_rate = slippage_bps / 10000.0

        self.use_bid_ask_spread = use_bid_ask_spread
        self.max_position = max_position
        self.initial_capital = initial_capital

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series | np.ndarray,
        ask_col: str = "ask_price_1",
        bid_col: str = "bid_price_1",
        mid_col: str = "mid_price",
    ) -> BacktestResult:
        """
        Execute backtest simulation over price DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Market data DataFrame containing order book prices.
        signals : pd.Series | np.ndarray
            Target position signal (-1, 0, 1).
        ask_col : str, default="ask_price_1"
            Best ask price column.
        bid_col : str, default="bid_price_1"
            Best bid price column.
        mid_col : str, default="mid_price"
            Mid price column.

        Returns
        -------
        BacktestResult
            Backtest equity curve, trade list, metrics, and position history.
        """
        n_samples = len(df)
        if n_samples == 0:
            raise ValueError("Cannot backtest empty DataFrame")

        signals_arr = np.asarray(signals, dtype=float)

        ask_prices = df[ask_col].to_numpy() if ask_col in df.columns else df[mid_col].to_numpy()
        bid_prices = df[bid_col].to_numpy() if bid_col in df.columns else df[mid_col].to_numpy()
        mid_prices = df[mid_col].to_numpy()

        cash = self.initial_capital
        current_position = 0.0
        entry_price = 0.0

        equity = np.zeros(n_samples, dtype=float)
        positions = np.zeros(n_samples, dtype=float)
        trades: list[dict[str, Any]] = []

        for t in range(n_samples):
            target_pos = np.clip(signals_arr[t], -self.max_position, self.max_position)
            current_mid = mid_prices[t]
            current_ask = ask_prices[t]
            current_bid = bid_prices[t]

            # Process position rebalancing / trade execution
            if target_pos != current_position:
                pos_change = target_pos - current_position

                if pos_change > 0:  # Buying (opening Long or closing Short)
                    base_price = current_ask if self.use_bid_ask_spread else current_mid
                    exec_price = base_price * (1.0 + self.slippage_rate)
                    notional = abs(pos_change) * exec_price
                    fee = notional * self.fee_rate
                    slippage_cost = abs(pos_change) * (exec_price - current_mid)

                    cash -= (notional + fee)

                    # Record trade PnL if closing position
                    trade_pnl = 0.0
                    if current_position < 0:
                        short_units = abs(current_position)
                        trade_pnl = short_units * (entry_price - exec_price) - fee

                    trades.append({
                        "event_idx": t,
                        "side": "BUY",
                        "units": abs(pos_change),
                        "exec_price": exec_price,
                        "mid_price": current_mid,
                        "fee": fee,
                        "slippage_cost": max(0.0, slippage_cost),
                        "pnl": trade_pnl,
                    })

                elif pos_change < 0:  # Selling (opening Short or closing Long)
                    base_price = current_bid if self.use_bid_ask_spread else current_mid
                    exec_price = base_price * (1.0 - self.slippage_rate)
                    notional = abs(pos_change) * exec_price
                    fee = notional * self.fee_rate
                    slippage_cost = abs(pos_change) * (current_mid - exec_price)

                    cash += (notional - fee)

                    # Record trade PnL if closing position
                    trade_pnl = 0.0
                    if current_position > 0:
                        long_units = current_position
                        trade_pnl = long_units * (exec_price - entry_price) - fee

                    trades.append({
                        "event_idx": t,
                        "side": "SELL",
                        "units": abs(pos_change),
                        "exec_price": exec_price,
                        "mid_price": current_mid,
                        "fee": fee,
                        "slippage_cost": max(0.0, slippage_cost),
                        "pnl": trade_pnl,
                    })

                current_position = target_pos
                entry_price = exec_price

            # Mark-to-market portfolio value at current mid price
            position_value = current_position * current_mid
            portfolio_value = cash + position_value

            equity[t] = portfolio_value
            positions[t] = current_position

        equity_series = pd.Series(equity, index=df.index, name="equity")
        position_series = pd.Series(positions, index=df.index, name="position")
        metrics = calculate_trading_metrics(equity_series, trades, self.initial_capital)

        return BacktestResult(
            equity_curve=equity_series,
            trades=trades,
            metrics=metrics,
            positions=position_series,
        )
