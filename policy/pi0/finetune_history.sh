#!/bin/bash
# sbatch -p sirius -N 1 -w byta6000i0 --gres=gpu:4 --job-name pi0 --wrap='bash finetune_history.sh pi0_base_aloha_robotwin_lora_history_transformer_fsdp4 put_back_block-history-transformer-fsdp4 0,1,2,3'
# bash finetune_history.sh pi0_base_aloha_robotwin_lora_history_transformer_fsdp4 put_back_block-history-transformer-fsdp4 0,1,2,3
set -euo pipefail

train_config_name="${1}"
exp_name="${2}"
gpu_id="${3:-0}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/bd_byt4090i0/users/zguan/huggingface_cache/huggingface/lerobot}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/bd_byt4090i0/users/zguan/huggingface_cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_history.py \
  "${train_config_name}" \
  --exp-name="${exp_name}" \
  --overwrite
