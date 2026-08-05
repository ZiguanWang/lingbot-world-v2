import argparse
from datetime import timedelta
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool


_I2V_EXAMPLE = {
    "prompt":
        "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder.",
    "image":
        "examples/04/image.jpg",
}

EXAMPLE_PROMPT = {
    "i2v-A14B": _I2V_EXAMPLE,
    "i2v-1.3B": _I2V_EXAMPLE,
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task.startswith("i2v"):
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="causal_fast",
        choices=["causal_fast", "causal_pretrain"],
        help="Inference mode: 'causal_fast' for the distilled few-step model, "
             "'causal_pretrain' for the pretrained causal model with 40-step CFG sampling.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4,
        help="The chunk size."),
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=None,
        help="Optional directory that holds shared T5 / VAE / tokenizer assets "
             "(used when the DiT checkpoint folder does not include them).")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--t5_rank0_cpu",
        action="store_true",
        default=False,
        help="Load/run CPU T5 only on rank 0 and broadcast embeddings to other ranks.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="The camera path to generate the video from.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--local_attn_size",
        type=int,
        default=-1,
        help='The local size of kv cache during inference')
    parser.add_argument(
        "--sink_size",
        type=int,
        default=0,
        help='The sink size of kv cache during inference')
    parser.add_argument(
        "--max_attention_size",
        type=int,
        default=None,
        help="The size of kv cache during inference.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default='output',
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=1,
        help="Number of generate() calls on the same loaded pipeline. Run 0 is cold; later runs are warm.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture a PyTorch CPU/CUDA trace for every benchmark run.")
    parser.add_argument(
        "--profile_dir",
        type=str,
        default="profiles",
        help="Directory for Chrome traces, operator tables, and benchmark summaries.")

    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def create_causal_pipeline(args, cfg, device, rank, mode="causal_fast"):
    logging.info(f"Creating WanI2VCausal pipeline (infer_mode={mode}).")
    return wan.WanI2VCausal(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        t5_rank0_cpu=args.t5_rank0_cpu,
        convert_model_dtype=args.convert_model_dtype,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode=mode,
        assets_dir=args.assets_dir,
    )


def run_causal(wan_i2v, args, img):
    logging.info("Generating video ...")
    return wan_i2v.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        chunk_size=args.chunk_size,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        seed=args.base_seed,
        offload_model=args.offload_model,
        max_attention_size=args.max_attention_size)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.t5_rank0_cpu and args.t5_fsdp:
        raise ValueError("--t5_rank0_cpu cannot be combined with --t5_fsdp")

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    cfg = WAN_CONFIGS[args.task]

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            # Rank 0 alone performs VAE decode.  At 77/361 frames this can
            # legitimately take longer than the 10-minute default while the
            # other ranks wait at the final collective.
            timeout=timedelta(hours=2))
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), "sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "The number of ulysses_size should be equal to the world size."
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
        init_distributed_group()

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    wan_i2v = create_causal_pipeline(
        args, cfg, device, rank, mode=args.infer_mode)

    if args.benchmark_runs < 1:
        raise ValueError("--benchmark_runs must be at least 1")
    run_results = []
    video = None
    for run_idx in range(args.benchmark_runs):
        run_kind = "cold" if run_idx == 0 else f"warm{run_idx}"
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        prof = None
        # A single rank is sufficient for the SP/FSDP critical-path trace.
        # Profiling every ROCm rank duplicates multi-GiB trace buffers and can
        # turn an otherwise valid inference into an OOM.
        if args.profile and rank == 0:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                # These two options retain tensor metadata/allocation events
                # for the whole run and add several GiB at 77+ frames.
                # Kernel timing and Chrome traces do not require them.
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            prof.__enter__()

        started_at = time.perf_counter()
        try:
            video = run_causal(wan_i2v, args, img)
            torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()
            # Deliberately stop wall-clock timing before profiler finalization,
            # Chrome trace export, and the potentially very slow key aggregation.
            dt = time.perf_counter() - started_at
        finally:
            if prof is not None:
                prof.__exit__(None, None, None)

        peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
        if rank == 0:
            logging.info(
                f"BENCHMARK run={run_kind} dt_s={dt:.6f} peak_allocated_gib={peak_gib:.3f}")

        if prof is not None:
            os.makedirs(args.profile_dir, exist_ok=True)
            prefix = os.path.join(
                args.profile_dir,
                f"{args.ulysses_size}gpu_{args.frame_num}f_"
                f"offload-{str(args.offload_model).lower()}_{run_kind}_rank{rank}")
            prof.export_chrome_trace(prefix + ".json")
            table_started_at = time.perf_counter()
            table = prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=200)
            table_dt = time.perf_counter() - table_started_at
            with open(prefix + "_ops.txt", "w", encoding="utf-8") as f:
                f.write(table)
                f.write(f"\nkey_averages_table_s={table_dt:.6f}\n")

        run_results.append((run_kind, dt, peak_gib))
        if run_idx + 1 < args.benchmark_runs:
            del video
            video = None

    if rank == 0:
        os.makedirs(args.profile_dir, exist_ok=True)
        summary_path = os.path.join(
            args.profile_dir,
            f"{args.ulysses_size}gpu_{args.frame_num}f_"
            f"offload-{str(args.offload_model).lower()}_summary.tsv")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("run\tdt_s\tpeak_allocated_gib\n")
            for run_kind, dt, peak_gib in run_results:
                f.write(f"{run_kind}\t{dt:.6f}\t{peak_gib:.3f}\n")

    if rank == 0 and video is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = '.mp4'
            actual_frames = int(video.shape[1])
            args.save_file = (
                f"lingbot-world-v2-{args.infer_mode}_"
                f"{args.size.replace('*','x') if sys.platform=='win32' else args.size}_"
                f"{args.ulysses_size}gpu_{args.frame_num}req_"
                f"{actual_frames}actual_{formatted_prompt}_{formatted_time}{suffix}"
            )
            args.save_file = f'{args.save_dir}/{args.save_file}'

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
    elif rank == 0:
        logging.info(
            "No in-process video returned; latent-only staged run completed.")

    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
