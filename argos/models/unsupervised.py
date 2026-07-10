"""
Unsupervised anomaly detection (IsolationForest + LOF ensemble).

This does NOT know what fraud looks like — it only knows what's
statistically unusual within the batch of transactions it's given.
Use this as a signal, or as the fallback when no trained supervised
model is available, but never present its output as a fraud
probability. Call it what it is: an anomaly score.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class UnsupervisedAnomalyScorer:
    def __init__(self, contamination: float = 0.02, n_neighbors: int = 20, random_state: int = 42):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns a 0..1 anomaly score per row (higher = more unusual).
        Requires at least a handful of rows to be meaningful — with
        very small batches (<10 rows) these models are not reliable,
        callers should surface that to the user.
        """
        n = X.shape[0]
        if n < 2:
            return np.zeros(n)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        iso = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        iso.fit(Xs)
        iso_scores = -iso.decision_function(Xs)

        n_neighbors = min(self.n_neighbors, max(1, n - 1))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=self.contamination)
        lof.fit_predict(Xs)
        lof_scores = -lof.negative_outlier_factor_

        mm = MinMaxScaler()
        combined_raw = np.vstack([iso_scores, lof_scores]).T
        combined_norm = mm.fit_transform(combined_raw)
        return combined_norm.mean(axis=1)
