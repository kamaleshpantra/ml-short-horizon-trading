import pandas as pd
import pytest

from trading_ml.data.parquet import (
    read_parquet,
    write_parquet,
)


def test_write_and_read_parquet(tmp_path):
    df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "symbol": ["TEST", "TEST", "TEST"],
            "bid_price_1": [100.0, 100.1, 100.2],
            "ask_price_1": [100.1, 100.2, 100.3],
        }
    )

    path = tmp_path / "test.parquet"

    written_path = write_parquet(df, path)

    assert written_path == path
    assert path.exists()

    loaded = read_parquet(path)

    pd.testing.assert_frame_equal(
        loaded,
        df,
    )


def test_write_parquet_creates_parent_directory(tmp_path):
    df = pd.DataFrame(
        {
            "value": [1, 2, 3],
        }
    )

    path = (
        tmp_path
        / "nested"
        / "directory"
        / "data.parquet"
    )

    write_parquet(df, path)

    assert path.exists()


def test_write_parquet_rejects_empty_dataframe(tmp_path):
    df = pd.DataFrame()

    path = tmp_path / "empty.parquet"

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        write_parquet(df, path)


def test_read_parquet_rejects_missing_file(tmp_path):
    path = tmp_path / "missing.parquet"

    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        read_parquet(path)