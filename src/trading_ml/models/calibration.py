from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


class CalibratedPredictor:
    """
    Wrapper for calibrating multi-class prediction probabilities using Platt Scaling (sigmoid)
    or Isotonic Regression.
    """

    def __init__(self, base_estimator: any, method: str = "sigmoid", cv: int = 3) -> None:
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.calibrator: CalibratedClassifierCV | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> CalibratedPredictor:
        """Fit probability calibrator over training/validation data."""
        if hasattr(self.base_estimator, "fit"):
            self.base_estimator.fit(X, y)

        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        raw_estimator = self.base_estimator.model if hasattr(self.base_estimator, "model") else self.base_estimator

        # Format labels if needed for XGBoost
        if hasattr(self.base_estimator, "_map_targets"):
            y_arr = self.base_estimator._map_targets(y_arr)

        self.calibrator = CalibratedClassifierCV(
            estimator=raw_estimator,
            method=self.method,
            cv=self.cv,
        )
        self.calibrator.fit(X_mat, y_arr)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return self.base_estimator.predict(X)
        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        raw_preds = self.calibrator.predict(X_mat)
        if hasattr(self.base_estimator, "_unmap_targets"):
            return self.base_estimator._unmap_targets(raw_preds)
        return raw_preds

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return self.base_estimator.predict_proba(X)
        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.calibrator.predict_proba(X_mat)
