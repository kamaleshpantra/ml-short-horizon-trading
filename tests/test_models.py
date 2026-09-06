import numpy as np
import pandas as pd
import pytest

from trading_ml.models.baseline import (
    LogisticRegressionBaseline,
    MajorityBaseline,
    OBIRuleBaseline,
)
from trading_ml.models.metrics import evaluate_classification
from trading_ml.models.xgboost_model import XGBoostMarketPredictor


def make_synthetic_dataset(n_samples: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    data = {
        "mid_price": 100.0 + np.cumsum(np.random.randn(n_samples) * 0.1),
        "spread": np.random.uniform(0.01, 0.1, n_samples),
        "relative_spread": np.random.uniform(0.0001, 0.001, n_samples),
        "obi_1": np.random.uniform(-0.8, 0.8, n_samples),
        "obi_3": np.random.uniform(-0.8, 0.8, n_samples),
        "obi_5": np.random.uniform(-0.8, 0.8, n_samples),
        "obi_10": np.random.uniform(-0.8, 0.8, n_samples),
        "microprice": 100.0 + np.random.randn(n_samples) * 0.1,
        "bid_depth_5": np.random.uniform(10, 100, n_samples),
        "ask_depth_5": np.random.uniform(10, 100, n_samples),
        "depth_ratio_5": np.random.uniform(0.5, 2.0, n_samples),
    }
    df = pd.DataFrame(data)
    y = pd.Series(np.random.choice([-1.0, 0.0, 1.0], size=n_samples, p=[0.3, 0.4, 0.3]))
    return df, y


def test_majority_baseline():
    X, y = make_synthetic_dataset(50)
    model = MajorityBaseline().fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == 50
    assert probs.shape == (50, 3)
    assert np.all(preds == model.majority_class)


def test_obi_rule_baseline():
    X, y = make_synthetic_dataset(50)
    model = OBIRuleBaseline(obi_col="obi_1", threshold=0.2).fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == 50
    assert probs.shape == (50, 3)
    assert set(np.unique(preds)).issubset({-1.0, 0.0, 1.0})


def test_logistic_regression_baseline():
    X, y = make_synthetic_dataset(60)
    model = LogisticRegressionBaseline(C=1.0).fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    coefs = model.get_coefficients()

    assert len(preds) == 60
    assert probs.shape == (60, 3)
    assert set(np.unique(preds)).issubset({-1.0, 0.0, 1.0})
    assert coefs.shape[0] == X.shape[1]


def test_xgboost_predictor():
    X, y = make_synthetic_dataset(60)
    model = XGBoostMarketPredictor(n_estimators=10, max_depth=2).fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    importances = model.get_feature_importances()

    assert len(preds) == 60
    assert probs.shape == (60, 3)
    assert set(np.unique(preds)).issubset({-1.0, 0.0, 1.0})
    assert len(importances) == X.shape[1]
    assert np.isclose(probs.sum(axis=1), 1.0).all()


def test_evaluate_classification_metrics():
    y_true = np.array([-1.0, 0.0, 1.0, 1.0, -1.0])
    y_pred = np.array([-1.0, 0.0, 1.0, 0.0, -1.0])
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.2, 0.6, 0.2],
        [0.9, 0.05, 0.05],
    ])

    metrics = evaluate_classification(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "f1_macro" in metrics
    assert "log_loss" in metrics
    assert metrics["accuracy"] == 4.0 / 5.0
