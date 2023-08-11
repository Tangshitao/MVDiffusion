#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH


n_nodes=1
n_gpus_per_node=1
torch_num_workers=0
batch_size=1
exp_name="test_mp3d=$(($n_gpus_per_node * $n_nodes * $batch_size))"

CUDA_VISIBLE_DEVICES='4' python -u ./test.py configs/pano_generation.yaml \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="cuda" --strategy="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} \
    --mode test \
    --ckpt_path weights/pano.ckpt
