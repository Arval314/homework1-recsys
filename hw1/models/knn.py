"""Item-based neighborhood collaborative filtering."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags


class ItemKNN:
    """Cosine item-to-item CF with overlap shrinkage and top-neighbor pruning."""

    def __init__(self, n_neighbors: int = 100, shrinkage: float = 20.0):
        self.n_neighbors = int(n_neighbors)
        self.shrinkage = float(shrinkage)

    def fit(self, positive_interactions: csr_matrix) -> "ItemKNN":
        self.history = csr_matrix(positive_interactions, dtype=np.float32)
        cooccurrence = (self.history.T @ self.history).tocsr()
        item_degree = np.asarray(cooccurrence.diagonal(), dtype=np.float32)
        inverse_norm = np.divide(
            1.0,
            np.sqrt(item_degree),
            out=np.zeros_like(item_degree, dtype=np.float32),
            where=item_degree > 0,
        )
        similarity = diags(inverse_norm) @ cooccurrence @ diags(inverse_norm)
        similarity = similarity.tocsr()
        similarity.setdiag(0)
        similarity.eliminate_zeros()

        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for item in range(similarity.shape[0]):
            start, stop = similarity.indptr[item : item + 2]
            neighbors = similarity.indices[start:stop]
            weights = similarity.data[start:stop]
            if self.shrinkage > 0 and len(neighbors):
                support = cooccurrence[item, neighbors].toarray().ravel()
                weights = weights * support / (support + self.shrinkage)
            keep = min(self.n_neighbors, len(neighbors))
            if keep == 0:
                continue
            selected = np.argpartition(weights, -keep)[-keep:]
            rows.extend([item] * keep)
            columns.extend(neighbors[selected].tolist())
            values.extend(weights[selected].astype(np.float32).tolist())
        self.similarity = csr_matrix(
            (values, (rows, columns)), shape=similarity.shape, dtype=np.float32
        )
        self.similarity.setdiag(0)
        self.similarity.eliminate_zeros()
        self.n_users, self.n_items = self.history.shape
        return self

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        scores = self.history[user_ids] @ self.similarity
        return scores.toarray().astype(np.float32, copy=False)
