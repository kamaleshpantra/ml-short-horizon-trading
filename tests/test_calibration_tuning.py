import numpy as np
import pandas as pd
import pytest

from trading_ml.models.calibration import CalibratedPredictor
from trading_ml.models.tuning import optimize_xgboost_hyperparameters
from trading_ml.models.xgboost_model import XGBoostMarketPredictor


def make_synthetic_dataset(n_samples: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    data = {
        "mid_price": 100.0 + np.cumsum(np.random.randn(n_samples) * 0.1),
        "spread": np.random.uniform(0.01, 0.1, n_samples),
        "obi_1": np.random.uniform(-0.8, 0.8, n_samples),
        "obi_5": np.random.uniform(-0.8, 0.8, n_samples),
        "microprice": 100.0 + np.random.randn(n_samples) * 0.1,
    }
    df = pd.DataFrame(data)
    y = pd.Series(np.random.choice([-1.0, 0.0, 1.0], size=n_samples, p=[0.3, 0.4, 0.3]))
    return df, y


def test_calibrated_predictor():
    X, y = make_synthetic_dataset(80)

    base_model = XGBoostMarketPredictor(n_estimators=10, max_depth=2)
    calibrated = CalibratedPredictor(base_model, method="sigmoid", cv=2).fit(X, y)

    preds = calibrated.predict(X)
    probs = calibrated.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 3)
    assert np.isclose(probs.sum(axis=1), 1.0).all()


def test_optimize_xgboost_hyperparameters():
    X, y = make_synthetic_dataset(100)
    df = X.copy()
    df["target"] = y

    param_grid = [
        {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 10},
        {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 10},
    ]

    best_params, best_score = optimize_xgboost_hyperparameters(
        df,
        feature_cols=["mid_price", "spread", "obi_1", "obi_5", "microprice"],
        target_col="target",
        n_splits=2,
        purge_window=5,
        param_grid=param_grid,
    )

    assert isinstance(best_params, dict)
    assert "max_depth" in best_params
    assert best_score >= 0.0
