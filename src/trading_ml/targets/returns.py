from __future__ import annotations

import pandas as pd


def add_future_return(
    df: pd.DataFrame,
    horizon: int = 10,
    price_col: str = "mid_price",
    return_col: str = "future_return",
) -> pd.DataFrame:
    """
    Calculate future return over a specified horizon H.

    Formula: r_{t,H} = (M_{t+H} - M_t) / M_t

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing market data.
    horizon : int, default=10
        Number of steps ahead into the future. Must be > 0.
    price_col : str, default="mid_price"
        Column name for price.
    return_col : str, default="future_return"
        Column name for the output future return.

    Returns
    -------
    pd.DataFrame
        DataFrame copy with future return column added. The final H rows will be NaN.
    """
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")

    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not found in DataFrame columns")

    result = df.copy()

    future_price = result[price_col].shift(-horizon)
    current_price = result[price_col]

    result[return_col] = (future_price - current_price) / current_price

    return result


def add_direction_target(
    df: pd.DataFrame,
    threshold: float = 0.0001,
    return_col: str = "future_return",
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Convert future returns into a discrete directional target label.

    Target classification:
        +1 : UP   (future_return > threshold)
        -1 : DOWN (future_return < -threshold)
         0 : HOLD (|future_return| <= threshold)
       NaN : Unknown future return (e.g. final H rows)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing future return column.
    threshold : float, default=0.0001
        Return magnitude threshold for direction classification. Must be > 0.
    return_col : str, default="future_return"
        Column name for future return.
    target_col : str, default="target"
        Column name for the output direction target.

    Returns
    -------
    pd.DataFrame
        DataFrame copy with direction target column added.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be a positive number, got {threshold}")

    if return_col not in df.columns:
        raise KeyError(f"Return column '{return_col}' not found in DataFrame columns")

    result = df.copy()
    ret = result[return_col]

    target = pd.Series(index=df.index, dtype=float)

    valid_mask = ret.notna()

    up_mask = valid_mask & (ret > threshold)
    down_mask = valid_mask & (ret < -threshold)
    hold_mask = valid_mask & (ret.abs() <= threshold)

    target[up_mask] = 1.0
    target[down_mask] = -1.0
    target[hold_mask] = 0.0

    result[target_col] = target

    return result


def build_targets(
    df: pd.DataFrame,
    horizon: int = 10,
    threshold: float = 0.0001,
    price_col: str = "mid_price",
    return_col: str = "future_return",
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Build future returns and directional targets in a single pipeline step.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price column.
    horizon : int, default=10
        Number of steps ahead into the future. Must be > 0.
    threshold : float, default=0.0001
        Threshold for classification. Must be > 0.
    price_col : str, default="mid_price"
        Column name for price.
    return_col : str, default="future_return"
        Column name for future return.
    target_col : str, default="target"
        Column name for target label.

    Returns
    -------
    pd.DataFrame
        DataFrame copy with future returns and target labels added.
    """
    df_returns = add_future_return(
        df,
        horizon=horizon,
        price_col=price_col,
        return_col=return_col,
    )

    df_targets = add_direction_target(
        df_returns,
        threshold=threshold,
        return_col=return_col,
        target_col=target_col,
    )

    return df_targets
