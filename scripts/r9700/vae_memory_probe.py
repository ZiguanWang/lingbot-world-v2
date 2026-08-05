import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from wan.modules.vae2_1 import Wan2_1_VAE
from wan.utils.utils import save_video


def stats(stage, tensor=None):
    torch.cuda.synchronize()
    free_b, total_b = torch.cuda.mem_get_info()
    fields = [
        f"VAE_PROBE stage={stage}",
        f"allocated_gib={torch.cuda.memory_allocated()/2**30:.3f}",
        f"reserved_gib={torch.cuda.memory_reserved()/2**30:.3f}",
        f"peak_gib={torch.cuda.max_memory_allocated()/2**30:.3f}",
        f"device_used_gib={(total_b-free_b)/2**30:.3f}",
    ]
    if tensor is not None:
        finite = int(torch.isfinite(tensor).sum())
        fields += [f"shape={tuple(tensor.shape)}", f"finite={finite}/{tensor.numel()}"]
    print(" ".join(fields), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("encode", "decode"), required=True)
    p.add_argument("--pixel_frames", type=int, default=13)
    p.add_argument("--latent_frames", type=int, default=4)
    p.add_argument("--height", type=int, default=464)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--explicit_sync", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--profile", action="store_true",
                   help="Profile only the VAE compute region.")
    p.add_argument("--profile_dir", default="profiles/vae",
                   help="Directory for the Chrome trace and operator tables.")
    p.add_argument("--reserve_gib", type=float, default=0.0)
    p.add_argument("--image")
    p.add_argument("--input", help="Load a CPU latent tensor for decode mode.")
    p.add_argument("--output", help="Save the resulting CPU tensor.")
    p.add_argument("--video_output", help="Save decode output as an MP4 video.")
    args = p.parse_args()

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = args.deterministic
    device = torch.device("cuda:0")
    print(
        f"VAE_CONFIG hip_launch_blocking={os.getenv('HIP_LAUNCH_BLOCKING')} "
        f"miopen_find_mode={os.getenv('MIOPEN_FIND_MODE')} "
        f"miopen_deterministic={os.getenv('MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC')} "
        f"torch_deterministic={args.deterministic} explicit_sync={args.explicit_sync} "
        f"dtype={args.dtype} "
        f"conv2d_split={os.getenv('LINGBOT_VAE_CONV2D_SPLIT', '0')}",
        flush=True,
    )
    vae_dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model_root = os.getenv("LINGBOT_MODEL_ROOT", "/mnt/raid0/roywan")
    ckpt_dir = os.getenv(
        "LINGBOT_CKPT_DIR",
        os.path.join(model_root, "lingbot-world-v2-14b-causal-fast"),
    )
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(ckpt_dir, "Wan2.1_VAE.pth"),
        dtype=vae_dtype,
        device=device,
    )
    reserve = None
    if args.reserve_gib:
        reserve = torch.empty(int(args.reserve_gib * 2**30), dtype=torch.uint8, device=device)
        reserve.fill_(1)
    stats("model_ready")
    torch.cuda.reset_peak_memory_stats()
    if args.mode == "encode":
        value = torch.zeros(3, args.pixel_frames, args.height, args.width,
                            dtype=torch.float32, device=device)
        if args.image:
            image = TF.to_tensor(Image.open(args.image).convert("RGB")).sub_(0.5).div_(0.5)
            value[:, 0] = torch.nn.functional.interpolate(
                image[None], size=(args.height, args.width), mode="bicubic")[0].to(device)
        else:
            value[:, 0].normal_()
    else:
        if args.input:
            value = torch.load(args.input, map_location=device, weights_only=True)
            if value.ndim != 4 or value.shape[0] != 16:
                raise ValueError(
                    f"Expected latent shape [16,T,H,W], got {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError("Input latent contains NaN/Inf")
            print(f"VAE_LOADED path={args.input} shape={tuple(value.shape)}", flush=True)
        else:
            value = torch.randn(16, args.latent_frames, args.height // 8, args.width // 8,
                                dtype=torch.float32, device=device)
    # Synchronize at both boundaries: HIP launches are asynchronous, so a CPU
    # timer without these fences measures launch latency instead of VAE work.
    torch.cuda.synchronize()
    started = time.perf_counter()
    prof = None
    try:
        if args.profile:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                # Retaining every Conv3D tensor shape makes ROCm traces grow
                # enough to stall or kill the profiler before export.
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            prof.__enter__()
        if args.mode == "encode":
            output = vae.encode([value])[0]
        else:
            output = vae.decode([value])[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        if prof is not None:
            prof.__exit__(None, None, None)
    stats(f"{args.mode}_done", output)
    print(f"VAE_COMPUTE_TIME seconds={elapsed:.6f}", flush=True)
    if prof is not None:
        os.makedirs(args.profile_dir, exist_ok=True)
        conv_tag = ("split2d" if
                    os.getenv("LINGBOT_VAE_CONV2D_SPLIT", "0") == "1"
                    else "native3d")
        prefix = os.path.join(
            args.profile_dir, f"vae17_{args.mode}_{args.dtype}_{conv_tag}")
        prof.export_chrome_trace(prefix + ".json")
        tables = {
            "_ops.txt": prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=200),
        }
        for suffix, table in tables.items():
            with open(prefix + suffix, "w", encoding="utf-8") as f:
                f.write(table)
                f.write("\n")
        print(f"VAE_PROFILE prefix={prefix}", flush=True)
    if args.output:
        if not torch.isfinite(output).all():
            raise RuntimeError("Refusing to save a non-finite VAE result")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        torch.save(output.detach().cpu(), args.output)
        print(f"VAE_SAVED path={args.output}", flush=True)
    if args.video_output:
        if args.mode != "decode":
            raise ValueError("--video_output is only supported for decode mode")
        if not torch.isfinite(output).all():
            raise RuntimeError("Refusing to save a video from non-finite VAE output")
        os.makedirs(os.path.dirname(os.path.abspath(args.video_output)), exist_ok=True)
        save_video(
            tensor=output.detach().cpu()[None],
            save_file=args.video_output,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        print(f"VAE_VIDEO_SAVED path={args.video_output}", flush=True)


if __name__ == "__main__":
    main()
