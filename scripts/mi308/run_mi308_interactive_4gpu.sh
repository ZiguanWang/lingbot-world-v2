#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"

export HIP_VISIBLE_DEVICES=${LINGBOT_MI308_GPUS:-4,5,6,7}
unset CUDA_VISIBLE_DEVICES || true
export LINGBOT_VAE_CONV2D_SPLIT=${LINGBOT_VAE_CONV2D_SPLIT:-1}
export LINGBOT_MEMORY_PROFILE=0

ckpt_dir=${LINGBOT_CKPT_DIR:-/mnt/raid0/roywan/lingbot-world-v2-14b-causal-fast}
port=${LINGBOT_INTERACTIVE_PORT:-7860}

exec torchrun --nproc_per_node=4 scripts/common/interactive_server.py \
  --ckpt_dir "$ckpt_dir" \
  --image "${LINGBOT_INTERACTIVE_IMAGE:-examples/03/image.jpg}" \
  --intrinsics "${LINGBOT_INTERACTIVE_INTRINSICS:-examples/03/intrinsics.npy}" \
  --max_area "${LINGBOT_INTERACTIVE_MAX_AREA:-99840}" \
  --chunk_size 4 \
  --ulysses_size 4 \
  --local_attn_size 18 \
  --sink_size 6 \
  --port "$port"
