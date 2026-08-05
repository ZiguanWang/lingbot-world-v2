#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"

out_dir="benchmarks/r9700/4gpu_native3d"
mkdir -p "$out_dir/logs" "$out_dir/videos"
video="$out_dir/videos/4gpu_33req_29actual.mp4"
log="$out_dir/logs/r9700_4gpu_33req_29actual.log"

unset HIP_LAUNCH_BLOCKING || true
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LINGBOT_FSDP_SYNC_MODULE_STATES=0
export LINGBOT_MEMORY_PROFILE=1
export LINGBOT_FSDP_PRE_WRAP_BARRIER=1
export LINGBOT_RUNTIME_WEIGHT_CHECK=0
unset LINGBOT_VAE_CONDITION_FILE || true
unset LINGBOT_PRED_LATENT_FILE || true
unset LINGBOT_SKIP_INPROCESS_VAE_DECODE || true
unset LINGBOT_STAGE_OFFLOAD_DIT_BEFORE_VAE || true

set +e
timeout --signal=TERM --kill-after=60s 30m \
    torchrun --nproc_per_node=4 scripts/common/generate.py \
    --task i2v-A14B \
    --infer_mode causal_fast \
    --size '480*832' \
    --frame_num 33 \
    --chunk_size 4 \
    --ckpt_dir lingbot-world-v2-14b-causal-fast \
    --ulysses_size 4 \
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
run_rc=${PIPESTATUS[0]}
set -e

printf 'FOUR_GPU_EXIT request=33 actual=29 rc=%s\n' "$run_rc" | tee -a "$log"
if [[ "$run_rc" -eq 0 && -s "$video" ]]; then
    python scripts/common/verify_generated_video.py \
        --expected-frames 29 "$video" | tee -a "$log"
fi

exit "$run_rc"
