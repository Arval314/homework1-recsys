"""Shared sparse-matrix helpers for the from-scratch recommenders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def positive_matrix(
    interactions: pd.DataFrame, n_users: int, n_items: int, threshold: float = 4.0
) -> csr_matrix:
    """Binary user-item matrix; only ratings at or above threshold are positive."""
    liked = interactions.loc[interactions["rating"] >= threshold]
    matrix = csr_matrix(
        (
            np.ones(len(liked), dtype=np.float32),
            (liked["user_id"].to_numpy(), liked["item_id"].to_numpy()),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    matrix.sort_indices()
    return matrix


def observed_matrix(
    interactions: pd.DataFrame, n_users: int, n_items: int
) -> csr_matrix:
    """Binary matrix of all observed reviews, positive and negative."""
    matrix = csr_matrix(
        (
            np.ones(len(interactions), dtype=np.float32),
            (interactions["user_id"].to_numpy(), interactions["item_id"].to_numpy()),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    matrix.sort_indices()
    return matrix
