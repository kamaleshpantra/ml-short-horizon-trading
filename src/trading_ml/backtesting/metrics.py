from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_trading_metrics(
    equity_curve: pd.Series,
    trades: list[dict[str, Any]],
    initial_capital: float = 100000.0,
) -> dict[str, float]:
    """
    Calculate comprehensive quantitative trading and risk metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Series of portfolio equity values through time.
    trades : list[dict[str, Any]]
        List of executed trade dictionaries.
    initial_capital : float, default=100000.0
        Starting portfolio capital.

    Returns
    -------
    dict[str, float]
        Dictionary of trading performance & risk metrics.
    """
    if len(equity_curve) == 0:
        return {}

    final_capital = float(equity_curve.iloc[-1])
    total_return = float((final_capital - initial_capital) / initial_capital)

    # Period returns
    returns = equity_curve.pct_change().fillna(0.0)

    # Risk metrics
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())
    sharpe_ratio = float(mean_ret / std_ret) if std_ret > 1e-8 else 0.0

    negative_returns = returns[returns < 0.0]
    downside_std = float(negative_returns.std()) if len(negative_returns) > 0 else 0.0
    sortino_ratio = float(mean_ret / downside_std) if downside_std > 1e-8 else 0.0

    # Drawdown series
    rolling_peak = equity_curve.cummax()
    drawdown = (rolling_peak - equity_curve) / rolling_peak
    max_drawdown = float(drawdown.max())

    # Trade statistics
    total_trades = len(trades)
    if total_trades > 0:
        trade_pnls = [t.get("pnl", 0.0) for t in trades]
        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p < 0]

        win_rate = float(len(winning_trades) / total_trades)
        gross_profit = float(sum(winning_trades))
        gross_loss = float(abs(sum(losing_trades)))

        profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-8 else (float("inf") if gross_profit > 0 else 0.0)
        total_fees = float(sum(t.get("fee", 0.0) for t in trades))
        total_slippage_cost = float(sum(t.get("slippage_cost", 0.0) for t in trades))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        total_fees = 0.0
        total_slippage_cost = 0.0

    return {
        "total_return": total_return,
        "final_capital": final_capital,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": float(total_trades),
        "total_fees": total_fees,
        "total_slippage_cost": total_slippage_cost,
    }
