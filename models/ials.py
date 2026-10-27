"""Confidence-weighted implicit ALS and a recency-quality target variant."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class ImplicitALS:
    """Hu-Koren-Volinsky style ALS over all user-item pairs.

    `signal="binary"` uses p_ui=1[r_ui>=4]. `signal="quality_recency"`
    uses a graded target p_ui=max(0,(r_ui-3)/2) * exp(-ln(2)*age/half_life).
    Both variants use the same observed-pair confidence c_ui=1+alpha.
    """

    def __init__(
        self,
        *,
        n_factors: int = 32,
        n_iters: int = 8,
        alpha: float = 20.0,
        reg: float = 0.1,
        signal: str = "binary",
        half_life_days: float = 365.0,
        random_state: int = 42,
    ):
        if signal not in {"binary", "quality_recency"}:
            raise ValueError("signal must be 'binary' or 'quality_recency'")
        if n_factors < 1 or n_iters < 1 or alpha < 0 or reg <= 0:
            raise ValueError("Require positive factors, iterations and regularization")
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.n_factors = int(n_factors)
        self.n_iters = int(n_iters)
        self.alpha = float(alpha)
        self.reg = float(reg)
        self.signal = signal
        self.half_life_days = float(half_life_days)
        self.random_state = int(random_state)

    def fit(
        self, interactions: pd.DataFrame, n_users: int, n_items: int
    ) -> "ImplicitALS":
        users = interactions["user_id"].to_numpy(dtype=np.int32)
        items = interactions["item_id"].to_numpy(dtype=np.int32)
        ratings = interactions["rating"].to_numpy(dtype=np.float32)
        timestamps = interactions["timestamp"].to_numpy(dtype=np.int64)

        rating_matrix = csr_matrix(
            (ratings, (users, items)), shape=(n_users, n_items), dtype=np.float32
        )
        rating_matrix.sort_indices()
        time_matrix = csr_matrix(
            (timestamps, (users, items)), shape=(n_users, n_items), dtype=np.int64
        )
        time_matrix.sort_indices()
        by_item = rating_matrix.tocsc()
        time_by_item = time_matrix.tocsc()
        if self.signal == "binary":
            targets = (rating_matrix.data >= 4.0).astype(np.float32)
        else:
            age_days = (timestamps.max() - time_matrix.data).astype(np.float64) / 86400.0
            strength = np.clip((rating_matrix.data.astype(np.float64) - 3.0) / 2.0, 0, 1)
            targets = (strength * np.exp(-np.log(2.0) * age_days / self.half_life_days)).astype(np.float32)

        preference_matrix = csr_matrix(
            (targets, rating_matrix.indices.copy(), rating_matrix.indptr.copy()),
            shape=rating_matrix.shape,
        )
        preference_by_item = preference_matrix.tocsc()
        self.train_matrix = rating_matrix
        self.user_factors = np.zeros((n_users, self.n_factors), dtype=np.float32)
        rng = np.random.default_rng(self.random_state)
        self.item_factors = rng.normal(
            0, 0.01, size=(n_items, self.n_factors)
        ).astype(np.float32)
        eye = np.eye(self.n_factors, dtype=np.float32) * self.reg
        confidence_delta = self.alpha
        confidence = 1.0 + self.alpha

        for _ in range(self.n_iters):
            item_gram = self.item_factors.T @ self.item_factors
            for user in range(n_users):
                start, stop = rating_matrix.indptr[user : user + 2]
                item_ids = rating_matrix.indices[start:stop]
                if len(item_ids) == 0:
                    continue
                factors = self.item_factors[item_ids]
                prefs = preference_matrix.data[start:stop]
                lhs = item_gram + confidence_delta * (factors.T @ factors) + eye
                rhs = confidence * (factors.T @ prefs)
                self.user_factors[user] = np.linalg.solve(lhs, rhs)

            user_gram = self.user_factors.T @ self.user_factors
            for item in range(n_items):
                start, stop = by_item.indptr[item : item + 2]
                user_ids = by_item.indices[start:stop]
                if len(user_ids) == 0:
                    self.item_factors[item] = 0
                    continue
                factors = self.user_factors[user_ids]
                prefs = preference_by_item.data[start:stop]
                lhs = user_gram + confidence_delta * (factors.T @ factors) + eye
                rhs = confidence * (factors.T @ prefs)
                self.item_factors[item] = np.linalg.solve(lhs, rhs)

        self.signal_values = targets
        self.n_users, self.n_items = n_users, n_items
        return self

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        return self.user_factors[user_ids] @ self.item_factors.T
