import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from typing import Union
from pathlib import Path
import os
import csv

DATASET_NAME = "ml-32m-filtered"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", REPO_ROOT / "data_preparation" / "filters" / "recsys26"))
DATASET_FOLDER = DATASET_ROOT / DATASET_NAME


def infer_ratings_csv_schema(ratings_path: Path) -> tuple[str, dict[str, str]]:
    with open(ratings_path, "r", encoding="utf-8", newline="") as f:
        header_line = f.readline().strip()
    if not header_line:
        raise ValueError(f"Ratings file is empty: {ratings_path}")

    separator = ","
    for candidate in [",", ";", "\t"]:
        if candidate in header_line:
            separator = candidate
            break

    columns = next(csv.reader([header_line], delimiter=separator))
    column_set = set(columns)
    rename_map: dict[str, str] = {}

    user_col = "userId" if "userId" in column_set else "user_id" if "user_id" in column_set else None
    item_col = "movieId" if "movieId" in column_set else "item_id" if "item_id" in column_set else None
    value_col = "rating" if "rating" in column_set else "value" if "value" in column_set else None
    if user_col is None or item_col is None or value_col is None:
        raise ValueError(
            f"Unsupported ratings schema in {ratings_path}. "
            f"Expected user/movie/rating columns, got: {columns}"
        )

    if user_col != "user_id":
        rename_map[user_col] = "user_id"
    if item_col != "item_id":
        rename_map[item_col] = "item_id"
    if value_col != "value":
        rename_map[value_col] = "value"

    return separator, rename_map


def load_interactions_dataframe(dataset_name: str) -> pl.DataFrame:
    if dataset_name != DATASET_NAME:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected '{DATASET_NAME}'.")
    ratings_path = DATASET_FOLDER / "ratings.csv"
    separator, rename_map = infer_ratings_csv_schema(ratings_path)
    interactions_df = (
        pl.scan_csv(str(ratings_path), separator=separator)
        .rename(rename_map)
        .select(["user_id", "item_id", "value"])
        .collect()
    )
    interactions_df = interactions_df.cast({"user_id": pl.String, "item_id": pl.String, "value": pl.Float32}).cast(
        {"user_id": pl.Categorical, "item_id": pl.Categorical}
    )
    print(f"Dataset info: users={interactions_df['user_id'].n_unique()}, items={interactions_df['item_id'].n_unique()}, interactions={len(interactions_df)}")
    return interactions_df


def convert_to_csr(interactions_df: pl.DataFrame) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    user_values = interactions_df["user_id"].cast(pl.String).to_numpy()
    item_values = interactions_df["item_id"].cast(pl.String).to_numpy()
    users, user_indices = np.unique(user_values, return_inverse=True)
    items, item_indices = np.unique(item_values, return_inverse=True)
    return (
        sp.csr_matrix(
            (
                np.ones(len(interactions_df), dtype=np.float32),
                (user_indices, item_indices),
            ),
            shape=(len(users), len(items)),
        ),
        users,
        items,
    )


def split_train_val_test_users(
    users, user_item_csr: sp.csr_matrix, val_ratio: float, test_ratio: float
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to < 1.")
    p = np.random.permutation(user_item_csr.shape[0])
    n_users = len(p)
    n_test = int(n_users * test_ratio)
    n_val = int(n_users * val_ratio)
    n_train = n_users - n_val - n_test
    train_user_idxs = p[:n_train]
    val_user_idxs = p[n_train : n_train + n_val]
    test_user_idxs = p[n_train + n_val :]
    train_csr, val_csr, test_csr = user_item_csr[train_user_idxs], user_item_csr[val_user_idxs], user_item_csr[test_user_idxs]
    train_users, val_users, test_users = users[train_user_idxs], users[val_user_idxs], users[test_user_idxs]
    print(f"Train split info: users={train_csr.shape[0]}, items={train_csr.shape[1]}, interactions={train_csr.nnz}")
    print(f"Val split info: users={val_csr.shape[0]}, items={val_csr.shape[1]}, interactions={val_csr.nnz}")
    print(f"Test split info: users={test_csr.shape[0]}, items={test_csr.shape[1]}, interactions={test_csr.nnz}")
    return train_csr, val_csr, test_csr, train_users, val_users, test_users


def prepare_interaction_data(cfg: dict) -> tuple[pl.DataFrame, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = cfg["dataset"]
    interactions_df = load_interactions_dataframe(dataset)
    interactions_csr, users, items = convert_to_csr(interactions_df)
    train_csr, val_csr, test_csr, train_users, val_users, test_users = split_train_val_test_users(
        users,
        interactions_csr,
        cfg["val_user_ratio"],
        cfg["test_user_ratio"],
    )
    return interactions_df, train_csr, val_csr, test_csr, train_users, val_users, test_users, items


def split_input_target_interactions(user_item_csr: sp.csr_matrix, target_ratio: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    if user_item_csr.shape[0] == 0:
        return user_item_csr.copy(), user_item_csr.copy()
    target_mask = np.concatenate(
        [
            np.random.permutation(np.array([True] * int(np.ceil(row_nnz * target_ratio)) + [False] * int((row_nnz - np.ceil(row_nnz * target_ratio)))))
            for row_nnz in np.diff(user_item_csr.indptr)
        ]
    )
    inputs, targets = user_item_csr.copy(), user_item_csr.copy()
    inputs.data *= ~target_mask
    targets.data *= target_mask
    inputs.eliminate_zeros()
    targets.eliminate_zeros()
    return inputs, targets


class Dataloader:
    def __init__(self, data: Union[sp.csr_matrix, np.ndarray, torch.Tensor], batch_size: int, device: torch.device, shuffle: bool = False):
        self.data = data
        self.dataset_size = self.data.shape[0]
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

    def __len__(self) -> int:
        return -(-self.dataset_size // self.batch_size)

    def __iter__(self):
        self.permutation = np.random.permutation(self.dataset_size) if self.shuffle else np.arange(self.dataset_size)
        self.i = 0
        return self

    def __next__(self) -> torch.Tensor:
        if self.i >= self.dataset_size:
            raise StopIteration
        next_i = min(self.i + self.batch_size, self.dataset_size)
        batch = self.data[self.permutation[self.i : next_i]]
        self.i = next_i
        return torch.tensor(batch.toarray() if isinstance(batch, sp.csr_matrix) else batch, device=self.device)
