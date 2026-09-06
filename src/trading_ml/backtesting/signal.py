from __future__ import annotations

import numpy as np
import pandas as pd


class SignalGenerator:
    """
    Signal Generator for mapping model prediction probabilities to directional target positions.

    Positions:
       +1 : Long
       -1 : Short
        0 : Flat
    """

    def __init__(
        self,
        buy_threshold: float = 0.50,
        sell_threshold: float = 0.50,
        prob_cols: tuple[str, str, str] = ("prob_down", "prob_hold", "prob_up"),
    ) -> None:
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.prob_cols = prob_cols

    def generate_signals(
        self,
        df: pd.DataFrame,
        probabilities: np.ndarray | None = None,
    ) -> pd.Series:
        """
        Generate trading signals from probability arrays or DataFrame probability columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        probabilities : np.ndarray | None, default=None
            Array of shape [N, 3] representing [P(DOWN), P(HOLD), P(UP)].

        Returns
        -------
        pd.Series
            Series of directional signals in {-1, 0, 1}.
        """
        if probabilities is not None:
            p_down = probabilities[:, 0]
            p_up = probabilities[:, 2]
        else:
            down_col, _, up_col = self.prob_cols
            if down_col not in df.columns or up_col not in df.columns:
                raise KeyError(f"Probability columns '{down_col}', '{up_col}' not found in DataFrame")
            p_down = df[down_col].to_numpy()
            p_up = df[up_col].to_numpy()

        signals = np.zeros(len(df), dtype=float)

        # Long signal when P(UP) > buy_threshold and P(UP) > P(DOWN)
        long_mask = (p_up > self.buy_threshold) & (p_up > p_down)
        signals[long_mask] = 1.0

        # Short signal when P(DOWN) > sell_threshold and P(DOWN) > P(UP)
        short_mask = (p_down > self.sell_threshold) & (p_down > p_up)
        signals[short_mask] = -1.0

        return pd.Series(signals, index=df.index, name="signal")
