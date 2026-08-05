#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"
request_frames=${1:?usage: $0 17|33|65|129|257}
case "$request_frames" in
  17) actual_frames=13 ;; 33) actual_frames=29 ;; 65) actual_frames=61 ;;
  129) actual_frames=125 ;; 257) actual_frames=253 ;;
  *) exit 2 ;;
esac
out_dir=benchmarks/r9700/4gpu_split2d
mkdir -p "$out_dir/logs" "$out_dir/videos"
video="$out_dir/videos/${request_frames}req_${actual_frames}actual.mp4"
log="$out_dir/logs/r9700_${request_frames}req_${actual_frames}actual.log"
unset HIP_LAUNCH_BLOCKING || true
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LINGBOT_VAE_CONV2D_SPLIT=1 LINGBOT_VAE_DTYPE=fp32
export LINGBOT_FSDP_SYNC_MODULE_STATES=0 LINGBOT_FSDP_PRE_WRAP_BARRIER=1
export LINGBOT_RUNTIME_WEIGHT_CHECK=0 LINGBOT_MEMORY_PROFILE=1
unset LINGBOT_VAE_CONDITION_FILE LINGBOT_PRED_LATENT_FILE || true
unset LINGBOT_SKIP_INPROCESS_VAE_DECODE LINGBOT_STAGE_OFFLOAD_DIT_BEFORE_VAE || true
set +e
timeout --signal=TERM --kill-after=60s 100m \
 torchrun --nproc_per_node=4 --no-python ./scripts/common/run_rank_numa_bound.sh \
 --task i2v-A14B --infer_mode causal_fast --size '480*832' \
 --frame_num "$request_frames" --chunk_size 4 \
 --ckpt_dir lingbot-world-v2-14b-causal-fast --ulysses_size 4 --dit_fsdp --t5_cpu \
 --image examples/03/image.jpg --action_path examples/03 --base_seed 42 \
 --prompt 'A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds.' \
 --local_attn_size 18 --sink_size 6 --benchmark_runs 1 \
 --profile_dir "$out_dir/profiles" --save_file "$video" \
 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}; set -e
printf 'SPLIT2D_4GPU_EXIT request=%s actual=%s rc=%s\n' "$request_frames" "$actual_frames" "$rc" | tee -a "$log"
if [[ $rc -eq 0 && -s $video ]]; then python scripts/common/verify_generated_video.py --expected-frames "$actual_frames" "$video" | tee -a "$log"; fi
exit "$rc"
