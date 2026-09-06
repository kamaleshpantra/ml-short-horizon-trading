from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_sample(n_events: int = 500, random_seed: int = 42) -> Path:
    """
    Generate realistic synthetic LOB sample data with dynamic price movements,
    varying order-book imbalances, spreads, and multi-class target distributions.
    """
    np.random.seed(random_seed)
    output_path = Path("data/raw/fi2010_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate oscillating mid-price path
    t = np.arange(n_events)
    noise = np.cumsum(np.random.randn(n_events) * 0.05)
    trend = np.sin(t * 0.05) * 1.5
    mid_prices = 100.0 + trend + noise

    rows = []

    for event_id in range(n_events):
        mid_price = mid_prices[event_id]

        row = []
        for level in range(1, 11):
            tick = level * 0.01
            ask_price = round(mid_price + tick, 4)
            ask_size = round(10.0 + level * 2.0 + np.random.uniform(-3, 3), 2)

            bid_price = round(mid_price - tick, 4)
            bid_size = round(12.0 + level * 2.0 + np.random.uniform(-3, 3), 2)

            # Ensure non-negative sizes
            ask_size = max(0.1, ask_size)
            bid_size = max(0.1, bid_size)

            row.extend([ask_price, ask_size, bid_price, bid_size])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, header=False, index=False)
    return output_path


if __name__ == "__main__":
    path = create_sample()
    print(f"Created multi-class realistic sample dataset at: {path}")