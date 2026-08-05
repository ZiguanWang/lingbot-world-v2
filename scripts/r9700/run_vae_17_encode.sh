#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"

dtype=${1:-fp32}
profile_mode=${2:-}
conv_mode=${3:-native3d}
case "$dtype" in fp32|bf16) ;; *) echo "usage: $0 [fp32|bf16] [profile] [native3d|split2d]" >&2; exit 2 ;; esac
case "$profile_mode" in
    "") profile_args=() ;;
    profile) profile_args=(--profile --profile_dir "benchmarks/r9700/vae17/profiles") ;;
    *) echo "second argument must be 'profile' or omitted" >&2; exit 2 ;;
esac
case "$conv_mode" in
    native3d) conv_env=0 ;;
    split2d) conv_env=1 ;;
    *) echo "third argument must be native3d or split2d" >&2; exit 2 ;;
esac

out_dir="benchmarks/r9700/vae17/${dtype}_encode_${conv_mode}"
mkdir -p "$out_dir"
taskset -c 0-7,64-71 env CUDA_VISIBLE_DEVICES=0 \
    LINGBOT_VAE_CONV2D_SPLIT="$conv_env" \
    python scripts/r9700/vae_memory_probe.py \
    --mode encode --pixel_frames 13 --height 464 --width 832 \
    --image examples/03/image.jpg --dtype "$dtype" \
    --output "$out_dir/condition.pt" "${profile_args[@]}" \
    2>&1 | tee "$out_dir/r9700_run${profile_mode:+_profile}.log"
