import numpy as np
import pandas as pd
import pytest

from trading_ml.targets.returns import (
    add_direction_target,
    add_future_return,
    build_targets,
)


def make_price_dataframe(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"mid_price": prices})


def test_add_future_return_calculation():
    # Price sequence: 100, 101, 102, 103, 104
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    df = make_price_dataframe(prices)

    # Horizon H = 2
    res = add_future_return(df, horizon=2)

    # For idx 0: (102 - 100) / 100 = 0.02
    # For idx 1: (103 - 101) / 101 = 2 / 101 ~ 0.01980198
    # For idx 2: (104 - 102) / 102 = 2 / 102 ~ 0.01960784
    # For idx 3: NaN (last 2 rows)
    # For idx 4: NaN
    assert res.loc[0, "future_return"] == pytest.approx(0.02)
    assert res.loc[1, "future_return"] == pytest.approx(2.0 / 101.0)
    assert res.loc[2, "future_return"] == pytest.approx(2.0 / 102.0)
    assert pd.isna(res.loc[3, "future_return"])
    assert pd.isna(res.loc[4, "future_return"])


def test_add_future_return_negative():
    prices = [100.0, 95.0, 90.0]
    df = make_price_dataframe(prices)

    res = add_future_return(df, horizon=1)

    # For idx 0: (95 - 100) / 100 = -0.05
    # For idx 1: (90 - 95) / 95 = -5 / 95
    # For idx 2: NaN
    assert res.loc[0, "future_return"] == pytest.approx(-0.05)
    assert res.loc[1, "future_return"] == pytest.approx(-5.0 / 95.0)
    assert pd.isna(res.loc[2, "future_return"])


def test_add_direction_target_classification():
    # Returns: 0.0005 (> 0.0001 -> UP +1), -0.0005 (< -0.0001 -> DOWN -1), 0.00005 (HOLD 0), 0.0 (HOLD 0), NaN (NaN)
    returns = [0.0005, -0.0005, 0.00005, 0.0, np.nan]
    df = pd.DataFrame({"future_return": returns})

    res = add_direction_target(df, threshold=0.0001)

    assert res.loc[0, "target"] == 1.0
    assert res.loc[1, "target"] == -1.0
    assert res.loc[2, "target"] == 0.0
    assert res.loc[3, "target"] == 0.0
    assert pd.isna(res.loc[4, "target"])


def test_last_h_observations_are_nan():
    # With horizon H = 3 on 10 rows, the final 3 rows must be NaN for both return and target
    prices = [100.0 + i for i in range(10)]
    df = make_price_dataframe(prices)

    res = build_targets(df, horizon=3, threshold=0.001)

    # First 7 rows have non-NaN returns and targets
    assert res["future_return"].iloc[:7].notna().all()
    assert res["target"].iloc[:7].notna().all()

    # Final 3 rows must be NaN (NOT zero / HOLD!)
    assert res["future_return"].iloc[-3:].isna().all()
    assert res["target"].iloc[-3:].isna().all()


def test_invalid_horizon():
    df = make_price_dataframe([100.0, 101.0])

    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        add_future_return(df, horizon=0)

    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        add_future_return(df, horizon=-5)

    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        add_future_return(df, horizon=1.5)  # type: ignore


def test_invalid_threshold():
    df = pd.DataFrame({"future_return": [0.01, -0.01]})

    with pytest.raises(ValueError, match="threshold must be a positive number"):
        add_direction_target(df, threshold=0.0)

    with pytest.raises(ValueError, match="threshold must be a positive number"):
        add_direction_target(df, threshold=-0.001)


def test_missing_price_column():
    df = pd.DataFrame({"other_column": [1, 2, 3]})

    with pytest.raises(KeyError, match="Price column 'mid_price' not found"):
        add_future_return(df, price_col="mid_price")


def test_missing_return_column():
    df = pd.DataFrame({"other_column": [1, 2, 3]})

    with pytest.raises(KeyError, match="Return column 'future_return' not found"):
        add_direction_target(df, return_col="future_return")


def test_dataframe_immutability():
    df = make_price_dataframe([100.0, 102.0, 104.0])
    original_cols = list(df.columns)
    original_values = df.copy()

    res = build_targets(df, horizon=1, threshold=0.001)

    # Ensure input df was not modified
    assert list(df.columns) == original_cols
    pd.testing.assert_frame_equal(df, original_values)

    # Result dataframe has new columns
    assert "future_return" in res.columns
    assert "target" in res.columns


def test_build_targets_integration():
    prices = [100.0, 100.2, 99.8, 100.0, 100.5]
    df = make_price_dataframe(prices)

    res = build_targets(df, horizon=2, threshold=0.001)

    assert "future_return" in res.columns
    assert "target" in res.columns
    assert len(res) == 5

    # Check row 0: (99.8 - 100.0)/100.0 = -0.002 < -0.001 -> DOWN (-1.0)
    assert res.loc[0, "future_return"] == pytest.approx(-0.002)
    assert res.loc[0, "target"] == -1.0

    # Check row 1: (100.0 - 100.2)/100.2 = -0.2 / 100.2 ~ -0.001996 < -0.001 -> DOWN (-1.0)
    assert res.loc[1, "target"] == -1.0

    # Check row 2: (100.5 - 99.8)/99.8 = 0.7 / 99.8 ~ 0.007014 > 0.001 -> UP (1.0)
    assert res.loc[2, "target"] == 1.0

    # Final 2 rows must be NaN
    assert pd.isna(res.loc[3, "target"])
    assert pd.isna(res.loc[4, "target"])
