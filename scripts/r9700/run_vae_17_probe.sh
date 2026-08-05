#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"

dtype=${1:-fp32}
case "$dtype" in
    fp32|bf16) ;;
    *) echo "usage: $0 [fp32|bf16] [latent.pt]" >&2; exit 2 ;;
esac

latent=${2:-benchmarks/r9700/staged_sync_async/latents/sync_17req_13actual.pt}
out_dir="benchmarks/r9700/vae17/${dtype}_standalone"
condition="$out_dir/condition.pt"
video="$out_dir/vae17.mp4"
mkdir -p "$out_dir"

# Keep the single-GPU process and its host allocations on NUMA node 0, which
# is the mapping used by GPU 0 on this host's production launch scripts.
runner=(taskset -c 0-7,64-71 env CUDA_VISIBLE_DEVICES=0)

"${runner[@]}" python scripts/r9700/vae_memory_probe.py \
    --mode encode --pixel_frames 13 --height 464 --width 832 \
    --image examples/03/image.jpg --dtype "$dtype" --output "$condition" \
    2>&1 | tee "$out_dir/r9700_encode.log"

"${runner[@]}" python scripts/r9700/vae_memory_probe.py \
    --mode decode --input "$latent" --height 464 --width 832 \
    --dtype "$dtype" --video_output "$video" \
    2>&1 | tee "$out_dir/r9700_decode.log"

python scripts/common/verify_generated_video.py --expected-frames 13 "$video" \
    | tee "$out_dir/r9700_verify.log"
sha256sum "$condition" "$video" | tee "$out_dir/sha256.txt"
printf 'VAE17_STANDALONE dtype=%s condition=%s latent=%s video=%s\n' \
    "$dtype" "$condition" "$latent" "$video"
