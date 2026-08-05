#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"

dtype=${1:-fp32}
latent=${2:-benchmarks/r9700/staged_sync_async/latents/sync_17req_13actual.pt}
profile_mode=${3:-}
conv_mode=${4:-native3d}
case "$dtype" in fp32|bf16) ;; *) echo "usage: $0 [fp32|bf16] [latent.pt] [profile] [native3d|split2d]" >&2; exit 2 ;; esac
case "$profile_mode" in
    "") profile_args=() ;;
    profile) profile_args=(--profile --profile_dir "benchmarks/r9700/vae17/profiles") ;;
    *) echo "third argument must be 'profile' or omitted" >&2; exit 2 ;;
esac
case "$conv_mode" in
    native3d) conv_env=0 ;;
    split2d) conv_env=1 ;;
    *) echo "fourth argument must be native3d or split2d" >&2; exit 2 ;;
esac
[[ -s "$latent" ]] || { echo "latent not found: $latent" >&2; exit 1; }

out_dir="benchmarks/r9700/vae17/${dtype}_decode_${conv_mode}"
mkdir -p "$out_dir"
video="$out_dir/vae17.mp4"
taskset -c 0-7,64-71 env CUDA_VISIBLE_DEVICES=0 \
    LINGBOT_VAE_CONV2D_SPLIT="$conv_env" \
    python scripts/r9700/vae_memory_probe.py \
    --mode decode --input "$latent" --height 464 --width 832 \
    --dtype "$dtype" --video_output "$video" "${profile_args[@]}" \
    2>&1 | tee "$out_dir/r9700_run${profile_mode:+_profile}.log"
python scripts/common/verify_generated_video.py --expected-frames 13 "$video" \
    | tee "$out_dir/r9700_verify${profile_mode:+_profile}.log"
