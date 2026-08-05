#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"

mode=${1:-}
request_frames=${2:-}
profile_mode=${3:-}

case "$mode" in
    sync) export HIP_LAUNCH_BLOCKING=1 ;;
    async) unset HIP_LAUNCH_BLOCKING || true ;;
    *) echo "usage: $0 {sync|async} {17|33|65|129|257} [profile]" >&2; exit 2 ;;
esac

case "$request_frames" in
    17) actual_frames=13 ;;
    33) actual_frames=29 ;;
    65) actual_frames=61 ;;
    129) actual_frames=125 ;;
    257) actual_frames=253 ;;
    *) echo "supported request frames: 17 33 65 129 257" >&2; exit 2 ;;
esac

profile_dir="benchmarks/r9700/staged_sync_async/profiles/${mode}_${request_frames}req_${actual_frames}actual"
case "$profile_mode" in
    "") run_suffix=""; profile_args=(--profile_dir "$profile_dir") ;;
    profile)
        run_suffix="_profile"
        profile_args=(
            --profile
            --profile_dir
            "$profile_dir"
        )
        ;;
    *) echo "third argument must be 'profile' or omitted" >&2; exit 2 ;;
esac

condition="benchmarks/r9700/staged_sync_async/conditions/vae_condition_${request_frames}req_${actual_frames}actual.pt"
latent="benchmarks/r9700/staged_sync_async/latents/${mode}_${request_frames}req_${actual_frames}actual${run_suffix}.pt"
video="benchmarks/r9700/staged_sync_async/videos/${mode}_${request_frames}req_${actual_frames}actual${run_suffix}.mp4"
dit_log="benchmarks/r9700/staged_sync_async/logs/r9700_${mode}_${request_frames}req_${actual_frames}actual${run_suffix}_dit.log"
decode_log="benchmarks/r9700/staged_sync_async/logs/r9700_${mode}_${request_frames}req_${actual_frames}actual${run_suffix}_decode.log"

mkdir -p benchmarks/r9700/staged_sync_async/conditions benchmarks/r9700/staged_sync_async/latents benchmarks/r9700/staged_sync_async/videos benchmarks/r9700/staged_sync_async/logs

# The image condition is independent of DiT scheduling. Reuse the validated
# finite tensor when present; otherwise create it in a clean one-GPU process.
if [[ ! -f "$condition" ]]; then
    CUDA_VISIBLE_DEVICES=0 python scripts/r9700/vae_memory_probe.py \
        --mode encode \
        --pixel_frames "$actual_frames" \
        --height 464 --width 832 \
        --image examples/03/image.jpg \
        --output "$condition"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LINGBOT_FSDP_SYNC_MODULE_STATES=0
export LINGBOT_MEMORY_PROFILE=1
export LINGBOT_SKIP_INPROCESS_VAE_DECODE=1
export LINGBOT_VAE_CONDITION_FILE="$condition"
export LINGBOT_PRED_LATENT_FILE="$latent"

torchrun --nproc_per_node=8 --no-python ./scripts/common/run_rank_numa_bound.sh \
    --task i2v-A14B \
    --infer_mode causal_fast \
    --size '480*832' \
    --frame_num "$request_frames" \
    --chunk_size 4 \
    --ckpt_dir lingbot-world-v2-14b-causal-fast \
    --ulysses_size 8 \
    --dit_fsdp \
    --t5_cpu \
    --image examples/03/image.jpg \
    --action_path examples/03 \
    --base_seed 42 \
    --prompt 'A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds.' \
    --local_attn_size 18 \
    --sink_size 6 \
    --benchmark_runs 1 \
    "${profile_args[@]}" \
    --save_file "benchmarks/r9700/staged_sync_async/videos/unused_${mode}_${request_frames}req_${actual_frames}actual.mp4" \
    2>&1 | tee "$dit_log"

reference_latent=""
reference_video=""
if [[ "$profile_mode" == profile ]]; then
    reference_latent="benchmarks/r9700/staged_sync_async/latents/sync_${request_frames}req_${actual_frames}actual.pt"
    reference_video="benchmarks/r9700/staged_sync_async/videos/sync_${request_frames}req_${actual_frames}actual.mp4"
elif [[ "$mode" == async ]]; then
    reference_latent="benchmarks/r9700/staged_sync_async/latents/sync_${request_frames}req_${actual_frames}actual.pt"
    reference_video="benchmarks/r9700/staged_sync_async/videos/sync_${request_frames}req_${actual_frames}actual.mp4"
fi

if [[ -n "$reference_latent" && -s "$reference_latent" && -s "$reference_video" ]] && \
        python scripts/common/compare_latents.py "$latent" "$reference_latent" | tee "$decode_log"; then
    # The VAE consumer receives exactly the same tensor, byte for byte. Reuse
    # the already decoded and verified video instead of spending up to 40
    # minutes recomputing an identical deterministic result.
    cp -- "$reference_video" "$video"
    printf 'VIDEO_REUSED exact_latent=true source=%s target=%s\n' \
        "$reference_video" "$video" | tee -a "$decode_log"
else
    # All FSDP/RCCL state is gone before this clean one-GPU VAE decode starts.
    CUDA_VISIBLE_DEVICES=0 python scripts/r9700/vae_memory_probe.py \
        --mode decode \
        --input "$latent" \
        --height 464 --width 832 \
        --video_output "$video" \
        2>&1 | tee "$decode_log"
fi

printf 'VIDEO_SAVED mode=%s request_frames=%s actual_frames=%s path=%s\n' \
    "$mode" "$request_frames" "$actual_frames" "$video"
