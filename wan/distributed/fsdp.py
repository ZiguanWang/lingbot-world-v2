import gc
import os
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    # In inference SHARD_GRAD_OP keeps successively unsharded transformer
    # blocks alive until the end of the forward.  The 14B causal DiT reaches
    # ~29.8 GiB/rank and OOMs on 32 GiB cards. FULL_SHARD reshards each block
    # after use; the isolated 8-GPU probe peaks at ~10.5 GiB/rank.
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    use_lora=False
):
    if os.getenv("LINGBOT_FSDP_SYNC_MODULE_STATES") is not None:
        sync_module_states = os.getenv("LINGBOT_FSDP_SYNC_MODULE_STATES") == "1"
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states,
        use_orig_params=True if use_lora else False)
    return model


def free_model(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()
