from typing import Dict, List
import os
import json

import numpy as np
import pandas as pd
from argdantic import ArgParser
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

from common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    input_csv: str = "dataset/refined_data.csv"
    output_dir: str = "data/user-trajectory-hrm"
    window_size: int = 5
    train_frac: float = 0.8
    seed: int = 42

def preprocess_dataframe(df: pd.DataFrame, config: DataProcessConfig):
    df = df.copy()

    # ---- Feature selection ----
    feature_cols = [
        "semantic_location",
        "hour",
        "day_of_week",
        "is_weekend",
        "wifi_status",
        "user",
    ]

    # Require timestamp for ordering
    df = df.dropna(subset=feature_cols + ["timestamp_long"])
    df = df.sort_values("timestamp_long").reset_index(drop=True)

    # ---- Encode features ----
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

    # ---- Assign vocab offsets ----
    offsets: Dict[str, int] = {}
    offset = 1  # 0 reserved for PAD
    for col in feature_cols:
        offsets[col] = offset
        offset += cardinalities[col]
    vocab_size = offset

    # ---- Build token matrix ----
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
    Build HRM tensors for one split.
    - Group by user
    - Sliding window per user
    """
    K = config.window_size

    results = {
        k: [] for k in [
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        ]
    }
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    puzzle_id = 0
    example_id = 0

    for user, df_u in df.groupby("user", sort=False):
        idx = df_u.index.to_numpy()
        token_seq = tokens[idx, :]
        n = token_seq.shape[0]

        if n < K:
            continue

        if user not in user_id_map:
            user_id_map[user] = len(user_id_map) + 1
        uid = user_id_map[user]

        for start in range(0, n - K + 1):
            window = token_seq[start:start + K, :]
            flat = window.reshape(-1)

            results["inputs"].append(flat)
            results["labels"].append(flat.copy())

            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(uid)

        results["group_indices"].append(puzzle_id)

    if example_id == 0:
        raise RuntimeError(f"No examples generated for split {set_name}")

    inputs = np.stack(results["inputs"], axis=0)
    labels = np.stack(results["labels"], axis=0)
    puzzle_indices = np.array(results["puzzle_indices"], dtype=np.int32)
    group_indices = np.array(results["group_indices"], dtype=np.int32)
    puzzle_identifiers = np.array(results["puzzle_identifiers"], dtype=np.int32)

    # ---- Sanity checks ----
    assert inputs.min() >= 0
    assert inputs.max() < vocab_size

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
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        format="mixed",
        errors="raise",
    )
    df["date"] = df["datetime"].dt.date

    # ---- DAY-BASED SPLIT PER USER ----
    train_parts = []
    test_parts = []

    for user, df_u in df.groupby("user", sort=False):
        days = sorted(df_u["date"].unique())
        if len(days) < 2:
            continue

        split_idx = int(np.ceil(config.train_frac * len(days)))
        train_days = set(days[:split_idx])
        test_days = set(days[split_idx:])

        train_parts.append(df_u[df_u["date"].isin(train_days)])
        test_parts.append(df_u[df_u["date"].isin(test_days)])

    df_train = pd.concat(train_parts).reset_index(drop=True)
    df_test = pd.concat(test_parts).reset_index(drop=True)

    # ---- Hard leakage check ----
    assert (
        df_train.merge(df_test, on=["user", "date"], how="inner").empty
    ), "Day leakage detected between train and test"

    # ---- Preprocess independently ----
    df_train, tokens_train, feature_cols, vocab_size_train, _, _ = preprocess_dataframe(df_train, config)
    df_test, tokens_test, _, vocab_size_test, _, _ = preprocess_dataframe(df_test, config)

    build_split(
        df=df_train,
        tokens=tokens_train,
        feature_cols=feature_cols,
        vocab_size=vocab_size_train,
        set_name="train",
        config=config,
        user_id_map={},
    )

    build_split(
        df=df_test,
        tokens=tokens_test,
        feature_cols=feature_cols,
        vocab_size=vocab_size_test,
        set_name="test",
        config=config,
        user_id_map={},
    )

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_dataset(config)

if __name__ == "__main__":
    cli()