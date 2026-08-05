import argparse
import gc
import os
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from wan.configs import WAN_CONFIGS
from wan.distributed.fsdp import shard_model
from wan.distributed.sequence_parallel import sp_attn_forward_causal, sp_dit_forward_causal
from wan.distributed.util import init_distributed_group
from wan.modules.model_fast import WanModelFast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="lingbot-world-v2-14b-causal-fast")
    parser.add_argument("--frame_num", type=int, default=361)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--local_attn_size", type=int, default=18)
    parser.add_argument("--text_len", type=int, default=512)
    parser.add_argument("--forwards", type=int, default=1)
    parser.add_argument(
        "--sharding",
        choices=("shard_grad_op", "full_shard"),
        default="shard_grad_op",
    )
    parser.add_argument("--sync_module_states", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stagger_wrap", action="store_true")
    return parser.parse_args()


def memory_row(stage, device):
    torch.cuda.synchronize(device)
    free_b, total_b = torch.cuda.mem_get_info(device)
    row = torch.tensor(
        [
            torch.cuda.memory_allocated(device),
            torch.cuda.memory_reserved(device),
            torch.cuda.max_memory_allocated(device),
            total_b - free_b,
            free_b,
        ],
        dtype=torch.float64,
        device=device,
    )
    rows = [torch.empty_like(row) for _ in range(dist.get_world_size())]
    dist.all_gather(rows, row)
    if dist.get_rank() == 0:
        values = [[round(v / 2**30, 3) for v in item.cpu().tolist()] for item in rows]
        print(
            f"DIT_MEMORY stage={stage} "
            "columns=allocated,reserved,peak_allocated,device_used,device_free "
            f"ranks={values}",
            flush=True,
        )
    dist.barrier()


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise ValueError("This probe is designed for exactly 8 ranks")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")
    init_distributed_group()
    memory_row("distributed_ready", device)

    mark_start = time.perf_counter()

    def mark(name):
        print(
            f"DIT_PROBE_MARK rank={rank} stage={name} "
            f"elapsed_s={time.perf_counter() - mark_start:.3f}",
            flush=True,
        )

    cfg = WAN_CONFIGS["i2v-A14B"]
    mark("checkpoint_load_start")
    model = WanModelFast.from_pretrained(
        args.ckpt_dir,
        subfolder=cfg.fast_checkpoint,
        torch_dtype=torch.bfloat16,
        local_attn_size=args.local_attn_size,
        sink_size=6,
    )
    if rank == 0:
        source_weight = model.blocks[20].ffn[2].weight.detach().cpu()
        print(
            "DIT_PARAM stage=pre_fsdp name=blocks.20.ffn.2.weight "
            f"shape={tuple(source_weight.shape)} dtype={source_weight.dtype} "
            f"finite={int(torch.isfinite(source_weight).sum())}/{source_weight.numel()} "
            f"absmax={float(torch.nan_to_num(source_weight.float()).abs().max())}",
            flush=True,
        )
    mark("checkpoint_load_done")
    model.eval().requires_grad_(False)
    mark("method_patch_start")
    for block in model.blocks:
        block.self_attn.forward = types.MethodType(sp_attn_forward_causal, block.self_attn)
    model.forward = types.MethodType(sp_dit_forward_causal, model)
    mark("method_patch_done")
    strategy = (
        ShardingStrategy.SHARD_GRAD_OP
        if args.sharding == "shard_grad_op"
        else ShardingStrategy.FULL_SHARD
    )
    mark("fsdp_wrap_start")
    if args.stagger_wrap:
        if args.sync_module_states:
            raise ValueError("--stagger_wrap requires --no-sync_module_states")
        for wrap_rank in range(world_size):
            if rank == wrap_rank:
                mark(f"fsdp_local_wrap_start_{wrap_rank}")
                model = shard_model(
                    model,
                    device_id=local_rank,
                    sharding_strategy=strategy,
                    sync_module_states=False,
                )
                mark(f"fsdp_local_wrap_done_{wrap_rank}")
            dist.barrier()
    else:
        model = shard_model(
            model,
            device_id=local_rank,
            sharding_strategy=strategy,
            sync_module_states=args.sync_module_states,
        )
    mark("fsdp_wrap_done")
    model = model.to(device)
    mark("model_to_device_done")
    dist.barrier()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    memory_row("model_sharded", device)

    lat_h, lat_w = 60, 104
    requested_lat_f = (args.frame_num - 1) // 4 + 1
    lat_f = requested_lat_f - requested_lat_f % args.chunk_size
    frame_seqlen = lat_h * lat_w // 4
    kv_size = frame_seqlen * args.local_attn_size
    local_heads = cfg.num_heads // world_size
    head_dim = cfg.dim // cfg.num_heads

    self_kv = []
    for _ in range(cfg.num_layers):
        self_kv.append(
            {
                "k": torch.zeros(1, kv_size, local_heads, head_dim, dtype=torch.bfloat16, device=device),
                "v": torch.zeros(1, kv_size, local_heads, head_dim, dtype=torch.bfloat16, device=device),
                "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
                "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
            }
        )
    memory_row("self_kv_ready", device)

    cross_kv = []
    for _ in range(cfg.num_layers):
        cross_kv.append(
            {
                "k": torch.zeros(1, args.text_len, cfg.num_heads, head_dim, dtype=torch.bfloat16, device=device),
                "v": torch.zeros(1, args.text_len, cfg.num_heads, head_dim, dtype=torch.bfloat16, device=device),
                "is_init": torch.tensor([False], dtype=torch.bool, device=device),
            }
        )
    memory_row("cross_kv_ready", device)

    camera = torch.zeros(1, 384, lat_f, lat_h, lat_w, dtype=torch.bfloat16, device=device)
    context = torch.zeros(args.text_len, 4096, dtype=torch.bfloat16, device=device)
    condition = torch.zeros(20, args.chunk_size, lat_h, lat_w, dtype=torch.bfloat16, device=device)
    latent = torch.zeros(16, args.chunk_size, lat_h, lat_w, dtype=torch.float32, device=device)
    timestep = torch.tensor([500.0], dtype=torch.float32, device=device)
    memory_row("forward_inputs_ready", device)

    def numeric_hook(name):
        def hook(_module, _inputs, output_value):
            value = output_value[0] if isinstance(output_value, (tuple, list)) else output_value
            if not torch.is_tensor(value):
                return
            count = torch.isfinite(value).sum()
            absmax = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0).abs().max()
            counts = [torch.empty_like(count) for _ in range(world_size)]
            maxima = [torch.empty_like(absmax) for _ in range(world_size)]
            dist.all_gather(counts, count)
            dist.all_gather(maxima, absmax)
            if rank == 0:
                print(
                    f"DIT_HOOK name={name} total={value.numel()} "
                    f"finite_per_rank={[int(x.item()) for x in counts]} "
                    f"absmax_per_rank={[float(x.item()) for x in maxima]}", flush=True)
        return hook

    model.module.patch_embedding.register_forward_hook(numeric_hook("patch_embedding"))
    model.module.time_embedding.register_forward_hook(numeric_hook("time_embedding"))
    model.module.text_embedding.register_forward_hook(numeric_hook("text_embedding"))
    model.module.blocks[0].register_forward_hook(numeric_hook("block_0"))
    model.module.blocks[1].register_forward_hook(numeric_hook("block_1"))
    for block_idx in range(10, 22):
        model.module.blocks[block_idx].register_forward_hook(
            numeric_hook(f"block_{block_idx}"))
    model.module.blocks[30].register_forward_hook(numeric_hook("block_30"))
    model.module.blocks[-1].register_forward_hook(numeric_hook("block_39"))
    model.module.head.register_forward_hook(numeric_hook("head"))
    suspect = model.module.blocks[20].module if isinstance(model.module.blocks[20], FSDP) else model.module.blocks[20]
    for sub_name in (
        "norm1", "self_attn", "cam_injector_layer1", "cam_injector_layer2",
        "cam_scale_layer", "cam_shift_layer", "norm3", "cross_attn", "norm2", "ffn",
    ):
        getattr(suspect, sub_name).register_forward_hook(
            numeric_hook(f"block20.{sub_name}"))
    for ffn_idx, ffn_layer in enumerate(suspect.ffn):
        ffn_layer.register_forward_hook(numeric_hook(f"block20.ffn.{ffn_idx}"))

    def ffn2_weight_pre_hook(module, _inputs):
        weight = module.weight.detach()
        finite = torch.isfinite(weight).sum()
        absmax = torch.nan_to_num(weight.float(), nan=0.0, posinf=0.0, neginf=0.0).abs().max()
        counts = [torch.empty_like(finite) for _ in range(world_size)]
        maxima = [torch.empty_like(absmax) for _ in range(world_size)]
        dist.all_gather(counts, finite)
        dist.all_gather(maxima, absmax)
        if rank == 0:
            print(
                "DIT_PARAM stage=forward_allgather name=blocks.20.ffn.2.weight "
                f"shape={tuple(weight.shape)} total={weight.numel()} "
                f"finite_per_rank={[int(x.item()) for x in counts]} "
                f"absmax_per_rank={[float(x.item()) for x in maxima]}", flush=True)

    suspect.ffn[2].register_forward_pre_hook(ffn2_weight_pre_hook)

    output = None
    for forward_idx in range(args.forwards):
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad(), model.no_sync(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                x=[latent],
                t=timestep,
                context=[context],
                seq_len=args.chunk_size * frame_seqlen,
                y=[condition],
                dit_cond_dict={"c2ws_plucker_emb": (camera[:, :, : args.chunk_size],)},
                kv_cache=self_kv,
                crossattn_cache=cross_kv,
                current_start=0,
                max_attention_size=kv_size,
                frame_seqlen=frame_seqlen,
                cross_attn_first_call=(forward_idx == 0),
            )[0]
        finite_count = torch.isfinite(output).sum()
        counts = [torch.empty_like(finite_count) for _ in range(world_size)]
        dist.all_gather(counts, finite_count)
        if rank == 0:
            print(
                f"DIT_NUMERIC forward={forward_idx} total={output.numel()} "
                f"finite_per_rank={[int(x.item()) for x in counts]}",
                flush=True,
            )
        memory_row(f"forward_{forward_idx}_done", device)

    del output, latent, condition, context, camera, self_kv, cross_kv, timestep
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    memory_row("inputs_and_caches_freed", device)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
