#!/bin/bash
set -euo pipefail

# History example:
# sbatch -p sirius -N 1 -w byt4090i0,byt4090i1 --gres=gpu:1 --job-name eval --wrap='bash eval.sh put_back_block demo_clean pi0_base_aloha_robotwin_lora_history_transformer_fsdp4 put_back_block-history-transformer-fsdp4 0 0 10000 8 put_back_block-demo_clean-50'
# bash eval.sh put_back_block demo_clean pi0_base_aloha_robotwin_lora_history_transformer_fsdp4 put_back_block-history-transformer-fsdp4 0 0 10000 50 put_back_block-demo_clean-50
# Keep evaluation independent from a preceding training phase.  The combined
# train+eval Slurm jobs set this variable to 0.95 for JAX training; inheriting
# that value leaves almost no VRAM for RoboTwin's Curobo planner processes.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
checkpoint_id=${7:-10000}
pi0_step=${8:-50}
# Baseline runs used model_name as the asset id. History runs have a distinct experiment name.
asset_id=${9:-${model_name}}
history_overflow=${10:-grow}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source "${script_dir}/.venv/bin/activate"
cd "${script_dir}/../.." # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config "policy/${policy_name}/deploy_policy.yml" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --train_config_name "${train_config_name}" \
    --model_name "${model_name}" \
    --asset_id "${asset_id}" \
    --checkpoint_id "${checkpoint_id}" \
    --pi0_step "${pi0_step}" \
    --history_overflow "${history_overflow}" \
    --ckpt_setting "${model_name}" \
    --checkpoint_num "${checkpoint_id}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
