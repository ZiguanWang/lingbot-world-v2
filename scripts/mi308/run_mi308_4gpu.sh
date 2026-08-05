#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"

request_frames=${1:?usage: $0 17|33|65|129|257}
case "$request_frames" in
    17|33|65|129|257) ;;
    *) echo "supported request frames: 17 33 65 129 257" >&2; exit 2 ;;
esac
chunk_size=${LINGBOT_CHUNK_SIZE:-4}
if [[ ! "$chunk_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "LINGBOT_CHUNK_SIZE must be a positive integer" >&2
    exit 2
fi
latent_frames=$(( (request_frames - 1) / 4 + 1 ))
usable_latent_frames=$(( latent_frames - latent_frames % chunk_size ))
if (( usable_latent_frames < 1 )); then
    echo "chunk_size=$chunk_size is too large for request_frames=$request_frames" >&2
    exit 2
fi
actual_frames=$(( (usable_latent_frames - 1) * 4 + 1 ))

# Override either the parent directory or the complete checkpoint path.
model_root=${LINGBOT_MODEL_ROOT:-/mnt/raid0/roywan}
ckpt_dir=${LINGBOT_CKPT_DIR:-${model_root}/lingbot-world-v2-14b-causal-fast}
gpu_devices=${LINGBOT_MI308_GPUS:-4,5,6,7}
nproc=${LINGBOT_MI308_NPROC:-4}
ulysses_size=${LINGBOT_MI308_ULYSSES_SIZE:-$nproc}
conv_mode=${LINGBOT_MI308_CONV_MODE:-native3d}
case "$conv_mode" in
    native3d) conv_split=0; default_out_dir=benchmarks/mi308/4gpu_native3d ;;
    split2d) conv_split=1; default_out_dir=benchmarks/mi308/4gpu_split2d ;;
    *) echo "LINGBOT_MI308_CONV_MODE must be native3d or split2d" >&2; exit 2 ;;
esac
out_dir=${LINGBOT_MI308_OUT_DIR:-$default_out_dir}
example_id=${LINGBOT_EXAMPLE_ID:-03}
case "$example_id" in
    00|01|02|03|04|05) ;;
    *) echo "LINGBOT_EXAMPLE_ID must be one of 00,01,02,03,04,05" >&2; exit 2 ;;
esac
example_dir="examples/$example_id"
if [[ -f "$example_dir/prompt.txt" ]]; then
    prompt=$(<"$example_dir/prompt.txt")
elif [[ "$example_id" == 03 ]]; then
    prompt='A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds.'
else
    prompt='A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder.'
fi
t5_device=${LINGBOT_MI308_T5_DEVICE:-cpu}
case "$t5_device" in
    cpu) t5_args=(--t5_cpu) ;;
    gpu) t5_args=() ;;
    *) echo "LINGBOT_MI308_T5_DEVICE must be cpu or gpu" >&2; exit 2 ;;
esac

if [[ ! -d "$ckpt_dir" ]]; then
    echo "checkpoint directory does not exist: $ckpt_dir" >&2
    exit 2
fi

mkdir -p "$out_dir/logs" "$out_dir/videos"
name_prefix=${LINGBOT_MI308_NAME_PREFIX:-${nproc}gpu_${request_frames}req_${actual_frames}actual_t5-${t5_device}}
video="$out_dir/videos/${name_prefix}.mp4"
log="$out_dir/logs/${name_prefix}.log"

# MI308 has enough HBM for one complete DiT copy per rank. Ulysses is kept for
# four-way sequence parallelism, but DiT FSDP and model offload are disabled.
export HIP_VISIBLE_DEVICES="$gpu_devices"
unset CUDA_VISIBLE_DEVICES || true
unset HIP_LAUNCH_BLOCKING || true
export LINGBOT_VAE_CONV2D_SPLIT="$conv_split"
unset LINGBOT_VAE_DTYPE || true
unset LINGBOT_VAE_CONDITION_FILE || true
unset LINGBOT_PRED_LATENT_FILE || true
unset LINGBOT_SKIP_INPROCESS_VAE_DECODE || true
unset LINGBOT_STAGE_OFFLOAD_DIT_BEFORE_VAE || true
export LINGBOT_MEMORY_PROFILE=${LINGBOT_MEMORY_PROFILE:-1}
export LINGBOT_RUNTIME_WEIGHT_CHECK=0

printf 'MI308_CONFIG physical_gpus=%s nproc=%s ulysses=%s chunk_size=%s ckpt_dir=%s t5_device=%s conv_mode=%s example=%s dit_fsdp=false offload=false\n' \
    "$gpu_devices" "$nproc" "$ulysses_size" "$chunk_size" "$ckpt_dir" "$t5_device" "$conv_mode" "$example_id" | tee "$log"

set +e
timeout --signal=TERM --kill-after=60s 90m \
    torchrun --nproc_per_node="$nproc" scripts/common/generate.py \
    --task i2v-A14B \
    --infer_mode causal_fast \
    --size '480*832' \
    --frame_num "$request_frames" \
    --chunk_size "$chunk_size" \
    --ckpt_dir "$ckpt_dir" \
    --ulysses_size "$ulysses_size" \
    "${t5_args[@]}" \
    --offload_model false \
    --image "$example_dir/image.jpg" \
    --action_path "$example_dir" \
    --base_seed 42 \
    --prompt "$prompt" \
    --local_attn_size 18 \
    --sink_size 6 \
    --benchmark_runs 1 \
    --profile_dir "$out_dir/profiles/t5-${t5_device}" \
    --save_file "$video" \
    2>&1 | tee -a "$log"
run_rc=${PIPESTATUS[0]}
set -e

printf 'MI308_GPU_EXIT nproc=%s request=%s actual=%s rc=%s\n' \
    "$nproc" "$request_frames" "$actual_frames" "$run_rc" | tee -a "$log"
if [[ "$run_rc" -eq 0 && -s "$video" ]]; then
    python scripts/common/verify_generated_video.py --expected-frames "$actual_frames" "$video" \
        | tee -a "$log"
fi
exit "$run_rc"
