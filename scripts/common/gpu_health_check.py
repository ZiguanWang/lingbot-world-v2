import argparse
import os
import time

import torch
import torch.distributed as dist


def sync_timed(label, fn, device):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = fn()
    torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--alloc-mib", type=int, default=1024)
    args = parser.parse_args()

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size > 1:
        dist.init_process_group("nccl")

    props = torch.cuda.get_device_properties(device)
    free_b, total_b = torch.cuda.mem_get_info(device)
    print(
        f"HEALTH rank={rank} device={local_rank} name={props.name} "
        f"free_mib={free_b / 2**20:.0f} total_mib={total_b / 2**20:.0f}",
        flush=True,
    )

    # Allocation and memory-bandwidth exercise.
    elements = args.alloc_mib * 2**20 // 4
    x, alloc_s = sync_timed(
        "allocate", lambda: torch.ones(elements, dtype=torch.float32, device=device), device
    )
    checksum = float(x[:: max(1, elements // 1024)].sum().item())
    print(
        f"HEALTH rank={rank} stage=allocate status=PASS dt_s={alloc_s:.4f} "
        f"checksum={checksum:.1f}", flush=True
    )
    del x
    torch.cuda.empty_cache()

    # Compute exercise.
    a = torch.randn((2048, 2048), dtype=torch.float16, device=device)
    b = torch.randn((2048, 2048), dtype=torch.float16, device=device)
    for iteration in range(args.iterations):
        c, matmul_s = sync_timed("matmul", lambda: a @ b, device)
        if not torch.isfinite(c).all():
            raise RuntimeError(f"rank {rank}: non-finite matmul output")
    print(
        f"HEALTH rank={rank} stage=matmul status=PASS iterations={args.iterations} "
        f"last_dt_s={matmul_s:.4f}", flush=True
    )
    del a, b, c

    # Conv3D exercise, matching the operation family that faulted in the VAE.
    conv = torch.nn.Conv3d(16, 64, kernel_size=3, padding=1).to(device, torch.float16)
    conv_input = torch.randn((1, 16, 8, 128, 128), dtype=torch.float16, device=device)
    for iteration in range(args.iterations):
        conv_out, conv_s = sync_timed("conv3d", lambda: conv(conv_input), device)
        if not torch.isfinite(conv_out).all():
            raise RuntimeError(f"rank {rank}: non-finite Conv3D output")
    print(
        f"HEALTH rank={rank} stage=conv3d status=PASS iterations={args.iterations} "
        f"last_dt_s={conv_s:.4f}", flush=True
    )
    del conv, conv_input, conv_out
    torch.cuda.empty_cache()

    # RCCL collectives exercise.
    if world_size > 1:
        collective = torch.full((16 * 2**20,), rank + 1, dtype=torch.float32, device=device)
        expected = world_size * (world_size + 1) / 2
        for iteration in range(args.iterations):
            collective.fill_(rank + 1)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            dist.all_reduce(collective)
            torch.cuda.synchronize(device)
            collective_s = time.perf_counter() - started
            if float(collective[0].item()) != expected:
                raise RuntimeError(f"rank {rank}: all_reduce checksum mismatch")
            dist.barrier()
        print(
            f"HEALTH rank={rank} stage=all_reduce status=PASS "
            f"iterations={args.iterations} last_dt_s={collective_s:.4f}", flush=True
        )

    print(f"HEALTH rank={rank} status=PASS", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
