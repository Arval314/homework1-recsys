"""Second-order Factorization Machine for ratings plus item metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


class FeatureEncoder:
    """Encode user, item, category, parent-category and brand feature IDs."""

    def __init__(self, n_users: int, n_items: int, metadata: pd.DataFrame, use_metadata: bool = True):
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.use_metadata = bool(use_metadata)
        self.metadata = metadata.sort_values("item_id").reset_index(drop=True)
        self._meta_by_item = self.metadata.set_index("item_id")
        self.category_map = self._vocabulary("category") if use_metadata else {}
        self.parent_map = self._vocabulary("parent_category") if use_metadata else {}
        self.brand_map = self._vocabulary("brand") if use_metadata else {}

        self.item_offset = self.n_users
        next_offset = self.item_offset + self.n_items
        self.category_offset = next_offset
        next_offset += len(self.category_map)
        self.parent_offset = next_offset
        next_offset += len(self.parent_map)
        self.brand_offset = next_offset
        next_offset += len(self.brand_map)
        self.n_features = next_offset

        item_meta = self._meta_by_item.reindex(np.arange(self.n_items))
        if use_metadata:
            self.category_ids = np.array(
                [self.category_map[value] + self.category_offset for value in item_meta["category"].fillna("Unknown category")],
                dtype=np.int32,
            )
            self.parent_ids = np.array(
                [self.parent_map[value] + self.parent_offset for value in item_meta["parent_category"].fillna("Unknown parent")],
                dtype=np.int32,
            )
            self.brand_ids = np.array(
                [self.brand_map[value] + self.brand_offset for value in item_meta["brand"].fillna("Unknown brand")],
                dtype=np.int32,
            )

    def _vocabulary(self, column: str) -> dict[str, int]:
        values = sorted(set(self.metadata[column].fillna("Unknown").astype(str)))
        return {value: index for index, value in enumerate(values)}

    def encode_pairs(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        users = np.asarray(user_ids, dtype=np.int32).reshape(-1)
        items = np.asarray(item_ids, dtype=np.int32).reshape(-1)
        columns = [users, self.item_offset + items]
        if self.use_metadata:
            columns.extend(
                [self.category_ids[items], self.parent_ids[items], self.brand_ids[items]]
            )
        return np.column_stack(columns).astype(np.int64, copy=False)


class _FMNetwork(nn.Module):
    def __init__(self, n_features: int, n_factors: int):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        self.latent = nn.Embedding(n_features, n_factors, sparse=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.02)

    def forward(self, feature_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        linear_weights = self.linear(feature_ids).squeeze(-1)
        linear = linear_weights.sum(dim=1)
        latent = self.latent(feature_ids)
        summed = latent.sum(dim=1)
        interaction = 0.5 * (summed.square() - latent.square().sum(dim=1)).sum(dim=1)
        return linear + interaction, latent


class FactorizationMachine:
    """FM trained with binary cross-entropy on observed 1–5 star ratings."""

    def __init__(
        self,
        *,
        n_factors: int = 16,
        epochs: int = 6,
        learning_rate: float = 0.03,
        reg: float = 1e-5,
        batch_size: int = 4096,
        use_metadata: bool = True,
        random_state: int = 42,
    ):
        self.n_factors = int(n_factors)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.reg = float(reg)
        self.batch_size = int(batch_size)
        self.use_metadata = bool(use_metadata)
        self.random_state = int(random_state)

    def fit(
        self,
        interactions: pd.DataFrame,
        n_users: int,
        n_items: int,
        metadata: pd.DataFrame,
    ) -> "FactorizationMachine":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.encoder = FeatureEncoder(n_users, n_items, metadata, self.use_metadata)
        self.network = _FMNetwork(self.encoder.n_features, self.n_factors)
        optimizer = torch.optim.SparseAdam(
            list(self.network.parameters()), lr=self.learning_rate
        )
        feature_ids = self.encoder.encode_pairs(
            interactions["user_id"].to_numpy(), interactions["item_id"].to_numpy()
        )
        labels = (interactions["rating"].to_numpy(dtype=np.float32) >= 4.0).astype(np.float32)
        rng = np.random.default_rng(self.random_state)
        self.loss_history: list[float] = []
        self.n_users, self.n_items = n_users, n_items

        for _ in range(self.epochs):
            order = rng.permutation(len(labels))
            epoch_loss = 0.0
            self.network.train()
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                x = torch.from_numpy(feature_ids[batch])
                y = torch.from_numpy(labels[batch])
                optimizer.zero_grad(set_to_none=True)
                logits, latent = self.network(x)
                linear_active = self.network.linear(x).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss = loss + 0.5 * self.reg * (
                    latent.square().sum(dim=(1, 2)).mean()
                    + linear_active.square().sum(dim=1).mean()
                )
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(batch)
            self.loss_history.append(epoch_loss / len(labels))
        return self

    def _predict_feature_ids(self, feature_ids: np.ndarray) -> np.ndarray:
        self.network.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(feature_ids), 65536):
                x = torch.from_numpy(feature_ids[start : start + 65536])
                logits, _ = self.network(x)
                chunks.append(logits.cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    def score_users(self, user_ids: np.ndarray, user_batch_size: int = 32) -> np.ndarray:
        users = np.asarray(user_ids, dtype=np.int32)
        all_items = np.arange(self.n_items, dtype=np.int32)
        result = np.empty((len(users), self.n_items), dtype=np.float32)
        for start in range(0, len(users), user_batch_size):
            batch_users = users[start : start + user_batch_size]
            pair_users = np.repeat(batch_users, self.n_items)
            pair_items = np.tile(all_items, len(batch_users))
            features = self.encoder.encode_pairs(pair_users, pair_items)
            result[start : start + len(batch_users)] = self._predict_feature_ids(features).reshape(
                len(batch_users), self.n_items
            )
        return result
