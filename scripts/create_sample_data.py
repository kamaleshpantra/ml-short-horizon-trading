from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_sample() -> Path:
    output_path = Path("data/raw/fi2010_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for event_id in range(100):
        mid_price = 100.0 + event_id * 0.01

        row = []

        for level in range(1, 11):
            tick = level * 0.01

            ask_price = mid_price + tick
            ask_size = 10.0 + level

            bid_price = mid_price - tick
            bid_size = 12.0 + level

            row.extend(
                [
                    ask_price,
                    ask_size,
                    bid_price,
                    bid_size,
                ]
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(
        output_path,
        header=False,
        index=False,
    )

    return output_path


if __name__ == "__main__":
    path = create_sample()
    print(f"Created sample dataset: {path}")