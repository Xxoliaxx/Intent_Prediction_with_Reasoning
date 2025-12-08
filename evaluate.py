from typing import List
import yaml
import os

import torch
import torch.distributed as dist
import numpy as np

import pydantic
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader


class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore
    
    RANK = 0
    WORLD_SIZE = 1
    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    # Dataloader
    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Models
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    # Try unwrap torch.compile
    try:
        train_state.model.load_state_dict(torch.load(eval_cfg.checkpoint, map_location="cuda"), assign=True)
    except:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in torch.load(eval_cfg.checkpoint, map_location="cuda").items()}, assign=True)
    
    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    print("Starting evaluation")

    train_state.model.eval()
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    # -------------------------------------------
    # CUSTOM METRIC: user + geohash accuracy only
    # -------------------------------------------
    try:
        # Load saved evaluation outputs from checkpoint directory
        ckpt_dir = os.path.dirname(eval_cfg.checkpoint)
        pred_file = None
        for f in os.listdir(ckpt_dir):
            if f.endswith("_all_preds.0"):
                pred_file = os.path.join(ckpt_dir, f)
                break

        if pred_file is None:
            raise RuntimeError("No prediction file found in checkpoint directory.")

        saved = torch.load(pred_file, map_location="cpu")

        logits = saved["logits"].to(torch.float32).cpu().numpy()
        labels = saved["labels"].to(torch.int64).cpu().numpy()


        preds = logits.argmax(-1)

        feature_cols = [
            "semantic_location",
            "hour",
            "day_of_week",
            "is_weekend",
            "wifi_status",
            "user",
        ]

        F = len(feature_cols)
        K = eval_metadata.seq_len // F

        target_features = ["semantic_location"]
        indices = []

        for f in target_features:
            fj = feature_cols.index(f)
            for step in range(K):
                indices.append(step * F + fj)

        indices = np.array(indices)

        pred_sub = preds[:, indices]
        label_sub = labels[:, indices]

        custom_acc = (pred_sub == label_sub).mean()
        metrics["custom_accuracy"] = float(custom_acc)

    except Exception as e:
        print("Failed computing custom accuracy:", e)

    # Print metrics including new key
    if metrics is not None:
        print(metrics)



if __name__ == "__main__":
    launch()
