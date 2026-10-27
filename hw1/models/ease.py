"""Embarrassingly Shallow Autoencoder, implemented with a dense linear solve."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csr_matrix


class EASE:
    def __init__(self, reg: float = 500.0):
        if reg <= 0:
            raise ValueError("reg must be positive")
        self.reg = float(reg)

    def fit(self, positive_interactions: csr_matrix) -> "EASE":
        self.history = csr_matrix(positive_interactions, dtype=np.float32)
        gram = (self.history.T @ self.history).toarray().astype(np.float64)
        gram.flat[:: gram.shape[0] + 1] += self.reg
        precision = solve(
            gram, np.eye(gram.shape[0]), assume_a="pos", check_finite=False
        )
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0.0)
        self.n_users, self.n_items = self.history.shape
        return self

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        scores = self.history[user_ids] @ self.weights
        return np.asarray(scores, dtype=np.float32)
