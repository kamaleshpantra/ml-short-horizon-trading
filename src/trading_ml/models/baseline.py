from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MajorityBaseline:
    """
    Baseline model that always predicts the majority class observed in the training set.
    """

    def __init__(self, classes: tuple[float, ...] = (-1.0, 0.0, 1.0)) -> None:
        self.classes = classes
        self.majority_class: float = 0.0

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> MajorityBaseline:
        y_arr = np.asarray(y)
        vals, counts = np.unique(y_arr, return_counts=True)
        self.majority_class = float(vals[np.argmax(counts)])
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        n_samples = len(X)
        return np.full(n_samples, self.majority_class, dtype=float)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        n_samples = len(X)
        proba = np.zeros((n_samples, len(self.classes)), dtype=float)
        try:
            class_idx = self.classes.index(self.majority_class)
            proba[:, class_idx] = 1.0
        except ValueError:
            proba[:, 1] = 1.0
        return proba


class OBIRuleBaseline:
    """
    Order-Book Imbalance heuristic baseline.

    Predicts:
       +1.0 (UP)   if obi > threshold
       -1.0 (DOWN) if obi < -threshold
        0.0 (HOLD) otherwise
    """

    def __init__(self, obi_col: str = "obi_1", threshold: float = 0.2) -> None:
        self.obi_col = obi_col
        self.threshold = threshold
        self.classes = (-1.0, 0.0, 1.0)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> OBIRuleBaseline:
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.obi_col not in X.columns:
                raise KeyError(f"OBI column '{self.obi_col}' not found in features DataFrame")
            obi = X[self.obi_col].to_numpy()
        else:
            obi = np.asarray(X)[:, 0]

        preds = np.zeros(len(obi), dtype=float)
        preds[obi > self.threshold] = 1.0
        preds[obi < -self.threshold] = -1.0
        return preds

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        n_samples = len(preds)
        proba = np.full((n_samples, 3), 0.2, dtype=float)

        proba[preds == -1.0, 0] = 0.6
        proba[preds == 0.0, 1] = 0.6
        proba[preds == 1.0, 2] = 0.6

        row_sums = proba.sum(axis=1, keepdims=True)
        return proba / row_sums


class LogisticRegressionBaseline:
    """
    Standardized Multi-Class Logistic Regression model for market prediction.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.classes_ = np.array([-1.0, 0.0, 1.0])

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> LogisticRegressionBaseline:
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.to_numpy()
        else:
            X_mat = np.asarray(X)

        y_arr = np.asarray(y, dtype=float)
        unique_classes = np.unique(y_arr)

        if len(unique_classes) < 2:
            self._single_class = float(unique_classes[0])
            return self

        self._single_class = None
        X_scaled = self.scaler.fit_transform(X_mat)
        self.model.fit(X_scaled, y_arr)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if getattr(self, "_single_class", None) is not None:
            return np.full(len(X), self._single_class, dtype=float)

        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.transform(X_mat)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if getattr(self, "_single_class", None) is not None:
            n_samples = len(X)
            proba = np.zeros((n_samples, 3), dtype=float)
            class_map = {-1.0: 0, 0.0: 1, 1.0: 2}
            class_idx = class_map.get(self._single_class, 1)
            proba[:, class_idx] = 1.0
            return proba

        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.transform(X_mat)

        raw_proba = self.model.predict_proba(X_scaled)
        n_samples = len(X_mat)
        full_proba = np.zeros((n_samples, len(self.classes_)), dtype=float)

        fitted_classes = self.model.classes_
        for idx, cls_val in enumerate(fitted_classes):
            matching_cols = np.where(self.classes_ == cls_val)[0]
            if len(matching_cols) > 0:
                full_proba[:, matching_cols[0]] = raw_proba[:, idx]

        return full_proba

    def get_coefficients(self) -> pd.DataFrame:
        """Return fitted model coefficients per feature and class."""
        if getattr(self, "_single_class", None) is not None or not hasattr(self.model, "coef_"):
            raise ValueError("Model is not fitted with multi-class data")

        feature_names = self.feature_names_ or [f"feat_{i}" for i in range(self.model.coef_.shape[1])]
        coef_df = pd.DataFrame(
            self.model.coef_.T,
            index=feature_names,
            columns=[f"coef_class_{c}" for c in self.model.classes_],
        )
        return coef_df
