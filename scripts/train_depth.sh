#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

n_nodes=1
n_gpus_per_node=4
torch_num_workers=8
batch_size=1
exp_name="train_depth=$(($n_gpus_per_node * $n_nodes * $batch_size))"

python -u ./train.py configs/depth_generation_train.yaml \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="cuda" --strategy="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} \
    --log_every_n_steps=100 \
    --limit_val_batches=1 \
    --num_sanity_val_steps=0 \
    --benchmark=True \
    --max_epochs=5 \
    --val_check_interval=0.5 \
    --gradient_clip_val=1.0 \
    --ckpt_path weights/depth_single_view.ckpt
