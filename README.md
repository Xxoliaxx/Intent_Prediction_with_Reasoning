# Intent Prediction with Hierarchical Reasoning (HRM)

This repository contains code, data pipelines, and experiment scripts for **intent prediction from user sensor data** using a **Hierarchical Reasoning Model (HRM)**.  
The project focuses on modeling **temporal user behavior** (location, device state, time context) and predicting **future semantic location and intent** using structured reasoning rather than shallow sequence modeling.

This work **extends an existing HRM framework** by adapting it to real-world mobile sensor data and designing a custom dataset construction pipeline.

## Environment Setup
Install Dependencies
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build Dataset
```
python dataset/build_trajectory_dataset.py --input-csv dataset/refined_data.csv --output-dir data/user-trajectory-hrm --window-size 5 --train-frac 0.8
```

## Training
```
OMP_NUM_THREADS=8 python pretrain.py data_path=data/user-trajectory-hrm epochs=2000 eval_interval=100 global_batch_size=384  lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

## Evaluation
```
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node 1 evaluate.py checkpoint="checkpoints/User-trajectory-hrm ACT-torch/HierarchicalReasoningModel_ACTV1 <checkpoint_name>/step_5200"
```
