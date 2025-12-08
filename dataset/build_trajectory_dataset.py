from typing import Dict, List
import os
import json

import numpy as np
import pandas as pd
from argdantic import ArgParser
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
import pygeohash as geohash

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    # Path to your CSV (relative to repo root)
    input_csv: str = "dataset/refined_data.csv"
    # Where to write HRM-ready data
    output_dir: str = "data/user-trajectory-hrm"
    # Number of timesteps per trajectory window
    window_size: int = 5
    # Train/test split fraction
    train_frac: float = 0.8
    seed: int = 42


def preprocess_dataframe(df: pd.DataFrame, config: DataProcessConfig):
    df = df.copy()

    # ---- 1) Parse lat/lon from "location" string ----
    # Example: "(17.4684131, 78.5714633)"
    loc = df["location"].astype(str).str.strip("()")
    lat = loc.str.split(",", expand=True)[0].astype(float)
    lon = loc.str.split(",", expand=True)[1].astype(float)

    # # Bucket lat/lon to keep vocab size reasonable
    # df["lat_bucket"] = lat.round(4).astype(str)
    # df["lon_bucket"] = lon.round(4).astype(str)

    # 1b) Compute geohash (MAIN CHANGE) ----
    df["geohash"] = [
        geohash.encode(lat_i, lon_i, precision=7)
        for lat_i, lon_i in zip(lat, lon)
    ]

    # ---- 2) Select features we'll feed into HRM ----
    feature_cols = [
        "semantic_location",
        "cluster",
        "hour",
        "day_of_week",
        "is_weekend",
        "wifi_status",
        "user",
    ]

    # Require timestamp_long for temporal ordering
    df = df.dropna(subset=feature_cols + ["timestamp_long"])
    df = df.sort_values("timestamp_long").reset_index(drop=True)

    # ---- 3) Label-encode each feature column ----
    encoders: Dict[str, LabelEncoder] = {}
    cardinalities: Dict[str, int] = {}
    encoded: Dict[str, np.ndarray] = {}

    for col in feature_cols:
        enc = LabelEncoder()
        vals = df[col].astype(str).fillna("<unk>")
        enc_vals = enc.fit_transform(vals)
        encoders[col] = enc
        cardinalities[col] = enc.classes_.shape[0]
        encoded[col] = enc_vals

    # ---- 4) Assign disjoint vocab ranges per feature ----
    # Token IDs are in [1, vocab_size-1], 0 = PAD
    offsets: Dict[str, int] = {}
    offset = 1
    for col in feature_cols:
        offsets[col] = offset
        offset += cardinalities[col]
    vocab_size = offset

    # ---- 5) Build token matrix (num_rows x num_features) ----
    num_rows = df.shape[0]
    num_features = len(feature_cols)
    tokens = np.empty((num_rows, num_features), dtype=np.int32)
    for j, col in enumerate(feature_cols):
        tokens[:, j] = offsets[col] + encoded[col]

    return df, tokens, feature_cols, vocab_size, encoders, offsets


def build_split(
    df: pd.DataFrame,
    tokens: np.ndarray,
    feature_cols: List[str],
    vocab_size: int,
    set_name: str,
    config: DataProcessConfig,
    user_id_map: Dict[str, int],
):
    """
    Build HRM tensors for one split (train or test).
    - Group by user
    - For each user, generate sliding windows of length K
    - Each window becomes one puzzle:
        inputs: flattened K x num_features tokens
        labels: same as inputs (reconstruction task)
    """
    K = config.window_size
    feature_cols = list(feature_cols)

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    puzzle_id = 0
    example_id = 0

    # Group by user; each user is a group
    for user, df_u in df.groupby("user", sort=False):
        idx = df_u.index.to_numpy()
        token_seq = tokens[idx, :]
        n = token_seq.shape[0]
        if n < K:
            continue

        # Assign a stable numeric ID per user
        if user not in user_id_map:
            user_id_map[user] = len(user_id_map) + 1
        uid = user_id_map[user]

        # Sliding window over this user's timeline
        for start in range(0, n - K + 1):
            window = token_seq[start : start + K, :]
            flat = window.reshape(-1)

            results["inputs"].append(flat)
            # For now, labels = inputs (reconstruction). This keeps HRM
            # training code unchanged. Later we can make this "next-step" prediction.
            results["labels"].append(flat.copy())

            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(uid)

        # One group per user
        results["group_indices"].append(puzzle_id)

    if example_id == 0:
        raise RuntimeError(f"No examples generated for split {set_name} (maybe too small or K too large)")

    inputs = np.stack(results["inputs"], axis=0)
    labels = np.stack(results["labels"], axis=0)
    puzzle_indices = np.array(results["puzzle_indices"], dtype=np.int32)
    group_indices = np.array(results["group_indices"], dtype=np.int32)
    puzzle_identifiers = np.array(results["puzzle_identifiers"], dtype=np.int32)

    # ---- Metadata ----
    metadata = PuzzleDatasetMetadata(
        seq_len=inputs.shape[1],
        vocab_size=vocab_size,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=int(puzzle_identifiers.max()) + 1,
        total_groups=len(group_indices) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    np.save(os.path.join(save_dir, "all__inputs.npy"), inputs)
    np.save(os.path.join(save_dir, "all__labels.npy"), labels)
    np.save(os.path.join(save_dir, "all__group_indices.npy"), group_indices)
    np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)

    df = pd.read_csv(config.input_csv)
    df, tokens, feature_cols, vocab_size, encoders, offsets = preprocess_dataframe(df, config)

    n = df.shape[0]
    split_idx = int(n * config.train_frac)

    df_train = df.iloc[:split_idx].reset_index(drop=True).copy()
    df_test = df.iloc[split_idx:].reset_index(drop=True).copy()

    tokens_train = tokens[:split_idx]
    tokens_test = tokens[split_idx:]

    user_id_map: Dict[str, int] = {}

    build_split(df_train, tokens_train, feature_cols, vocab_size, "train", config, user_id_map)
    build_split(df_test, tokens_test, feature_cols, vocab_size, "test", config, user_id_map)

    # Optional: save user ID mapping for inspection
    inv_map = {v: k for k, v in user_id_map.items()}
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([inv_map.get(i, "<blank>") for i in range(max(user_id_map.values()) + 1)], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()