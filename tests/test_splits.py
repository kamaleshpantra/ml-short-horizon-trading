import numpy as np
import pandas as pd
import pytest

from trading_ml.validation.splits import (
    PurgedWalkForwardCV,
    temporal_train_val_test_split,
)


def make_sample_df(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame({"event_id": np.arange(rows), "value": np.random.randn(rows)})


def test_temporal_train_val_test_split_ratios():
    df = make_sample_df(100)
    train, val, test = temporal_train_val_test_split(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_window=5,
        embargo_window=0,
    )

    # Raw bounds: Train 0..70, Val 70..85, Test 85..100
    # Purging 5 before boundary:
    # Train: 0..65 (65 rows)
    # Val: 70..80 (10 rows)
    # Test: 85..100 (15 rows)
    assert len(train) == 65
    assert len(val) == 10
    assert len(test) == 15

    # Check temporal ordering
    assert train["event_id"].max() < val["event_id"].min()
    assert val["event_id"].max() < test["event_id"].min()


def test_temporal_split_purging_and_embargo():
    df = make_sample_df(100)
    train, val, test = temporal_train_val_test_split(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_window=10,
        embargo_window=2,
    )

    # Raw bounds: Train 0..70, Val 70..85, Test 85..100
    # Purging 10 before boundary & Embargo 2 after boundary:
    # Train: 0..(70-10) = 0..60 (60 rows)
    # Val: (70+2)..(85-10) = 72..75 (3 rows)
    # Test: (85+2)..100 = 87..100 (13 rows)
    assert len(train) == 60
    assert val["event_id"].min() == 72
    assert test["event_id"].min() == 87


def test_temporal_split_invalid_ratios():
    df = make_sample_df(100)

    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        temporal_train_val_test_split(df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)

    with pytest.raises(ValueError, match="strictly positive"):
        temporal_train_val_test_split(df, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)


def test_purged_walk_forward_cv_splits():
    df = make_sample_df(100)
    cv = PurgedWalkForwardCV(
        n_splits=3,
        purge_window=5,
        embargo_window=1,
        min_train_ratio=0.4,
    )

    folds = list(cv.split(df))
    assert len(folds) == 3

    for train_idx, val_idx in folds:
        assert len(train_idx) > 0
        assert len(val_idx) > 0

        # Purge & embargo guarantee train indices strictly precede val indices with gap
        train_max = train_idx.max()
        val_min = val_idx.min()
        assert train_max < val_min


def test_purged_walk_forward_cv_invalid_args():
    df = make_sample_df(10)

    with pytest.raises(ValueError, match="n_splits must be positive"):
        PurgedWalkForwardCV(n_splits=0)

    with pytest.raises(ValueError, match="Dataset too small"):
        cv = PurgedWalkForwardCV(n_splits=10, min_train_ratio=0.5)
        list(cv.split(df))
