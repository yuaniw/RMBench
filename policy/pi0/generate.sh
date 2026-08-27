#!/bin/bash

set -euo pipefail

data_dir="${1}"
repo_id="${2}"
seed="${3:-0}"

uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir "${data_dir}" \
  --repo-id "${repo_id}" \
  --seed "${seed}"
