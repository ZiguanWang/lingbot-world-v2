#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"

variant=${1:?usage: $0 VARIANT}
actual_frames=29
condition="benchmarks/r9700/staged_sync_async/conditions/vae_condition_33req_29actual.pt"
out_dir="benchmarks/r9700/ablations/async33"
mkdir -p "$out_dir/logs" "$out_dir/latents" "$out_dir/videos"

unset HIP_LAUNCH_BLOCKING || true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LINGBOT_FSDP_SYNC_MODULE_STATES=0
export LINGBOT_MEMORY_PROFILE=1
export LINGBOT_FSDP_PRE_WRAP_BARRIER=1
unset LINGBOT_RUNTIME_WEIGHT_CHECK || true
unset LINGBOT_STAGE_OFFLOAD_DIT_BEFORE_VAE || true

launcher=(torchrun --nproc_per_node=8 --no-python ./scripts/common/run_rank_numa_bound.sh)
export LINGBOT_VAE_CONDITION_FILE="$condition"
export LINGBOT_PRED_LATENT_FILE="$out_dir/latents/${variant}.pt"
export LINGBOT_SKIP_INPROCESS_VAE_DECODE=1

case "$variant" in
    no_numa_binding)
        launcher=(torchrun --nproc_per_node=8 scripts/common/generate.py)
        ;;
    no_pre_wrap_barrier)
        export LINGBOT_FSDP_PRE_WRAP_BARRIER=0
        ;;
    rank0_weight_stats)
        export LINGBOT_RUNTIME_WEIGHT_CHECK=1
        ;;
    inline_vae_condition)
        unset LINGBOT_VAE_CONDITION_FILE
        ;;
    inprocess_decode_saved_latent)
        unset LINGBOT_SKIP_INPROCESS_VAE_DECODE
        export LINGBOT_STAGE_OFFLOAD_DIT_BEFORE_VAE=1
        ;;
    inprocess_decode_no_latent_file)
        unset LINGBOT_SKIP_INPROCESS_VAE_DECODE
        unset LINGBOT_PRED_LATENT_FILE
        export LINGBOT_STAGE_OFFLOAD_DIT_BEFORE_VAE=1
        ;;
    inprocess_decode_keep_dit)
        unset LINGBOT_SKIP_INPROCESS_VAE_DECODE
        ;;
    *)
        echo "unknown variant: $variant" >&2
        exit 2
        ;;
esac

video="$out_dir/videos/${variant}.mp4"
log="$out_dir/logs/r9700_${variant}.log"

timeout --signal=TERM --kill-after=60s 30m "${launcher[@]}" \
    --task i2v-A14B \
    --infer_mode causal_fast \
    --size '480*832' \
    --frame_num 33 \
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
    --profile_dir "$out_dir/profiles" \
    --save_file "$video" \
    2>&1 | tee "$log"

if [[ -s "$out_dir/latents/${variant}.pt" ]]; then
    python scripts/common/compare_latents.py \
        "$out_dir/latents/${variant}.pt" \
        benchmarks/r9700/staged_sync_async/latents/sync_33req_29actual.pt | tee -a "$log"
fi

if [[ -s "$video" ]]; then
    python scripts/common/verify_generated_video.py "$video" --expected-frames "$actual_frames" | tee -a "$log"
fi
