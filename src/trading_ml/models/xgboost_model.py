from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


class XGBoostMarketPredictor:
    """
    XGBoost Gradient Boosting Classifier for multi-class market direction prediction.

    Target Mapping:
        -1.0 (DOWN) -> 0
         0.0 (HOLD) -> 1
         1.0 (UP)   -> 2
    """

    LABEL_MAP = {-1.0: 0, 0.0: 1, 1.0: 2}
    INV_LABEL_MAP = {0: -1.0, 1: 0.0, 2: 1.0}

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
        )
        self.feature_names_: list[str] = []

    def _map_targets(self, y: pd.Series | np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        mapped = np.zeros(len(y_arr), dtype=int)
        for label_val, idx in self.LABEL_MAP.items():
            mapped[y_arr == label_val] = idx
        return mapped

    def _unmap_targets(self, y_mapped: np.ndarray) -> np.ndarray:
        unmapped = np.zeros(len(y_mapped), dtype=float)
        for idx, label_val in self.INV_LABEL_MAP.items():
            unmapped[y_mapped == idx] = label_val
        return unmapped

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        eval_set: list[tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]] | None = None,
    ) -> XGBoostMarketPredictor:
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.to_numpy()
        else:
            X_mat = np.asarray(X)

        y_mapped = self._map_targets(y)

        formatted_eval_set = None
        if eval_set is not None:
            formatted_eval_set = []
            for eval_X, eval_y in eval_set:
                eval_X_mat = eval_X.to_numpy() if isinstance(eval_X, pd.DataFrame) else np.asarray(eval_X)
                eval_y_mapped = self._map_targets(eval_y)
                formatted_eval_set.append((eval_X_mat, eval_y_mapped))

        self.model.fit(
            X_mat,
            y_mapped,
            eval_set=formatted_eval_set,
            verbose=False,
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_pred_mapped = self.model.predict(X_mat)
        return self._unmap_targets(y_pred_mapped)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        raw_proba = self.model.predict_proba(X_mat)
        n_samples = len(X_mat)
        full_proba = np.zeros((n_samples, 3), dtype=float)

        fitted_classes = getattr(self.model, "classes_", np.array([0, 1, 2]))
        if len(raw_proba.shape) == 2 and raw_proba.shape[1] == 3:
            return raw_proba

        for idx, cls_idx in enumerate(fitted_classes):
            if 0 <= cls_idx < 3:
                full_proba[:, int(cls_idx)] = raw_proba[:, idx]
        return full_proba

    def get_feature_importances(self) -> pd.Series:
        """Return feature importances as a pandas Series sorted descending."""
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model is not fitted yet")

        importances = self.model.feature_importances_
        feature_names = self.feature_names_ or [f"feat_{i}" for i in range(len(importances))]
        series = pd.Series(importances, index=feature_names, name="importance")
        return series.sort_values(ascending=False)
