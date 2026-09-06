from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.models.metrics import evaluate_classification
from trading_ml.models.xgboost_model import XGBoostMarketPredictor
from trading_ml.validation.splits import PurgedWalkForwardCV

logger = logging.getLogger(__name__)


def optimize_xgboost_hyperparameters(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    n_splits: int = 3,
    purge_window: int = 10,
    param_grid: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Hyperparameter optimization for XGBoost using Purged Walk-Forward Cross Validation.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame containing features and target.
    feature_cols : list[str]
        List of feature column names.
    target_col : str, default="target"
        Target column name.
    n_splits : int, default=3
        Number of walk-forward CV splits.
    purge_window : int, default=10
        Purge window prior to validation fold.
    param_grid : list[dict[str, Any]] | None, default=None
        Custom list of hyperparameter dictionary candidates.

    Returns
    -------
    tuple[dict[str, Any], float]
        (best_params, best_log_loss)
    """
    if param_grid is None:
        param_grid = [
            {"max_depth": 2, "learning_rate": 0.01, "n_estimators": 50, "subsample": 0.8},
            {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 50, "subsample": 0.8},
            {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 100, "subsample": 0.7},
            {"max_depth": 5, "learning_rate": 0.10, "n_estimators": 100, "subsample": 0.8},
        ]

    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_window=purge_window, min_train_ratio=0.4)

    best_score = float("inf")
    best_params = param_grid[0]

    for params in param_grid:
        fold_scores = []

        for train_idx, val_idx in cv.split(df):
            train_sub = df.iloc[train_idx]
            val_sub = df.iloc[val_idx]

            X_tr, y_tr = train_sub[feature_cols], train_sub[target_col]
            X_va, y_va = val_sub[feature_cols], val_sub[target_col]

            model = XGBoostMarketPredictor(**params)
            model.fit(X_tr, y_tr)

            probs = model.predict_proba(X_va)
            preds = model.predict(X_va)
            metrics = evaluate_classification(y_va, preds, probs)

            loss = metrics.get("log_loss", float("inf"))
            if not np.isnan(loss):
                fold_scores.append(loss)

        avg_loss = float(np.mean(fold_scores)) if fold_scores else float("inf")
        logger.info("Tested params %s -> Avg CV Log Loss: %.5f", params, avg_loss)

        if avg_loss < best_score:
            best_score = avg_loss
            best_params = params

    logger.info("Best XGBoost Hyperparameters: %s (CV Log Loss: %.5f)", best_params, best_score)
    return best_params, best_score
