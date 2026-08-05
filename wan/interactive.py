"""Persistent, chunk-at-a-time interactive inference for LingBot World v2."""

import math
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from einops import rearrange
from scipy.spatial.transform import Rotation

from .utils.cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
)


def action_to_poses(action, chunk_size=4):
    """Convert a browser input snapshot to one chunk of OpenCV c2w poses."""
    keys = set(action.get("keys", []))
    mouse_dx = float(action.get("mouse_dx", 0.0))
    mouse_dy = float(action.get("mouse_dy", 0.0))
    yaw = np.clip(mouse_dx * 0.0025, -0.45, 0.45)
    pitch = np.clip(mouse_dy * 0.0025, -0.35, 0.35)
    move = np.array([
        float("d" in keys) - float("a" in keys),
        float("e" in keys) - float("q" in keys),
        float("w" in keys) - float("s" in keys),
    ], dtype=np.float32)
    if np.linalg.norm(move) > 0:
        move /= np.linalg.norm(move)

    poses = np.repeat(np.eye(4, dtype=np.float32)[None], chunk_size, axis=0)
    denom = max(chunk_size - 1, 1)
    for i in range(chunk_size):
        alpha = i / denom
        rot = Rotation.from_euler("yx", [yaw * alpha, pitch * alpha]).as_matrix()
        poses[i, :3, :3] = rot.astype(np.float32)
        poses[i, :3, 3] = (move * alpha).astype(np.float32)
    return poses


class InteractiveSession:
    """Keep model/cache state alive and advance one causal chunk per action."""

    def __init__(self, pipe, prompt, image, intrinsics, *, max_area,
                 chunk_size=4, shift=10.0, seed=42,
                 max_attention_size=None):
        if pipe.infer_mode != "causal_fast":
            raise ValueError("InteractiveSession requires infer_mode='causal_fast'")
        self.pipe = pipe
        self.rank = pipe.rank
        self.device = pipe.device
        self.chunk_size = chunk_size
        self.max_attention_size = max_attention_size
        self.step_index = 0
        self.closed = False
        self.seed_g = torch.Generator(device=self.device).manual_seed(seed)

        img = TF.to_tensor(image).sub_(0.5).div_(0.5).to(self.device)
        aspect_ratio = img.shape[1] / img.shape[2]
        self.lat_h = round(
            np.sqrt(max_area * aspect_ratio) // pipe.vae_stride[1] //
            pipe.patch_size[1] * pipe.patch_size[1])
        self.lat_w = round(
            np.sqrt(max_area / aspect_ratio) // pipe.vae_stride[2] //
            pipe.patch_size[2] * pipe.patch_size[2])
        self.height = self.lat_h * pipe.vae_stride[1]
        self.width = self.lat_w * pipe.vae_stride[2]
        self.frame_seqlen = self.lat_h * self.lat_w // 4
        self.max_seq_len = math.ceil(
            chunk_size * self.lat_h * self.lat_w /
            (pipe.patch_size[1] * pipe.patch_size[2]) / pipe.sp_size
        ) * pipe.sp_size

        pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=shift)
        self.timesteps = pipe.scheduler.timesteps[[0, 179, 358, 679]]

        cache_key = __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest()
        if cache_key in pipe._t5_cache:
            self.context = pipe._t5_cache[cache_key]
        elif not pipe.t5_cpu:
            pipe.text_encoder.model.to(self.device)
            self.context = pipe.text_encoder([prompt], self.device)
            pipe._t5_cache[cache_key] = self.context
        else:
            context = pipe.text_encoder([prompt], torch.device("cpu"))
            self.context = [x.to(self.device) for x in context]
            pipe._t5_cache[cache_key] = self.context

        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)
        if intrinsics.ndim == 1:
            intrinsics = intrinsics[None]
        self.intrinsics = get_Ks_transformed(
            intrinsics, height_org=480, width_org=832,
            height_resize=self.height, width_resize=self.width,
            height_final=self.height, width_final=self.width)[0]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32) + 0.5,
            torch.arange(self.width, device=self.device, dtype=torch.float32) + 0.5,
            indexing="ij")
        fx, fy, cx, cy = self.intrinsics.to(self.device)
        self.base_camera_dirs = torch.stack([
            (grid_x.reshape(-1) - cx) / fx,
            (grid_y.reshape(-1) - cy) / fy,
            torch.ones(self.height * self.width, device=self.device),
        ], dim=-1)
        self.base_camera_dirs /= self.base_camera_dirs.norm(dim=-1, keepdim=True)

        first_rgb_frames = (chunk_size - 1) * pipe.vae_stride[0] + 1
        if self.rank == 0:
            initial_video = torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(self.height, self.width),
                    mode="bicubic").transpose(0, 1),
                torch.zeros(3, first_rgb_frames - 1, self.height, self.width),
            ], dim=1).to(self.device)
            condition = pipe.vae.encode([initial_video])[0]
        else:
            condition = torch.empty(
                16, chunk_size, self.lat_h, self.lat_w,
                dtype=torch.float32, device=self.device)
        if dist.is_initialized():
            if self.rank == 0:
                torch.cuda.synchronize(self.device)
            dist.broadcast(condition, src=0)
        mask = torch.zeros(
            4, chunk_size, self.lat_h, self.lat_w,
            dtype=torch.float32, device=self.device)
        mask[:, 0] = 1
        self.first_condition = torch.concat([mask, condition])
        self.empty_condition = torch.zeros_like(self.first_condition)

        model_args = pipe.model.config
        head_dim = model_args.dim // model_args.num_heads
        local_heads = model_args.num_heads // pipe.sp_size
        kv_frames = pipe.local_attn_size if pipe.local_attn_size > -1 else 1024
        self.kv_size = self.frame_seqlen * kv_frames
        self.self_kv_cache = pipe._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=[1, self.kv_size, local_heads, head_dim],
            dtype=pipe.pipe_dtype, device=self.device)
        self.cross_kv_cache = pipe._initialize_crossattn_cache(
            num_layers=model_args.num_layers,
            shape=[1, 512, model_args.num_heads, head_dim],
            dtype=pipe.pipe_dtype, device=self.device)
        pipe._cross_attn_initialized = False
        if self.rank == 0:
            pipe.vae.begin_decode_stream()

    def _camera_embedding(self, poses):
        poses = torch.as_tensor(poses, dtype=torch.float32)
        if poses.shape != (self.chunk_size, 4, 4):
            raise ValueError(
                f"Expected poses {(self.chunk_size, 4, 4)}, got {tuple(poses.shape)}")
        poses = compute_relative_poses(poses, framewise=True).to(self.device)
        rays_d = (poses[:, :3, :3] @ self.base_camera_dirs.T).transpose(1, 2)
        rays_o = poses[:, :3, 3].unsqueeze(1).expand_as(rays_d)
        emb = torch.cat([rays_o, rays_d], dim=-1).view(
            self.chunk_size, self.height, self.width, 6)
        emb = rearrange(
            emb, "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=self.height // self.lat_h, c2=self.width // self.lat_w)
        emb = rearrange(
            emb[None], "b (f h w) c -> b c f h w",
            f=self.chunk_size, h=self.lat_h, w=self.lat_w)
        return emb.to(self.pipe.param_dtype)

    @torch.no_grad()
    def step(self, action):
        if self.closed:
            raise RuntimeError("InteractiveSession is closed")
        started = time.perf_counter()
        poses = action_to_poses(action, self.chunk_size)
        camera = self._camera_embedding(poses)
        torch.cuda.synchronize(self.device)
        camera_s = time.perf_counter() - started
        dit_started = time.perf_counter()
        current = torch.randn(
            16, self.chunk_size, self.lat_h, self.lat_w,
            dtype=torch.float32, generator=self.seed_g, device=self.device)
        condition = self.first_condition if self.step_index == 0 else self.empty_condition
        kwargs = {
            "context": [self.context[0]],
            "seq_len": self.max_seq_len,
            "y": [condition],
            "dit_cond_dict": {"c2ws_plucker_emb": camera.chunk(1, dim=0)},
            "kv_cache": self.self_kv_cache,
            "crossattn_cache": self.cross_kv_cache,
            "current_start": self.step_index * self.chunk_size * self.frame_seqlen,
            "max_attention_size": self.kv_size if self.max_attention_size is None
                                  else self.max_attention_size,
            "frame_seqlen": self.frame_seqlen,
        }
        no_sync = getattr(self.pipe.model, "no_sync", nullcontext)
        with torch.amp.autocast("cuda", dtype=self.pipe.param_dtype), no_sync():
            for idx, timestep_value in enumerate(self.timesteps):
                timestep = torch.stack([timestep_value]).to(self.device)
                pred = self.pipe.model(
                    x=[current], t=timestep,
                    cross_attn_first_call=not self.pipe._cross_attn_initialized,
                    **kwargs)[0]
                self.pipe._cross_attn_initialized = True
                x0 = self.pipe._convert_flow_pred_to_x0(
                    flow_pred=pred, xt=current, timestep=timestep_value,
                    scheduler=self.pipe.scheduler)
                if idx + 1 < len(self.timesteps):
                    current = self.pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=self.seed_g,
                                        device=x0.device, dtype=x0.dtype),
                        self.timesteps[idx + 1])
            zero_t = torch.stack([self.timesteps[-1] * 0.0]).to(self.device)
            self.pipe.model(
                x=[x0], t=zero_t, cross_attn_first_call=False, **kwargs)
        torch.cuda.synchronize(self.device)
        dit_s = time.perf_counter() - dit_started

        frames = None
        decode_s = 0.0
        if self.rank == 0:
            decode_started = time.perf_counter()
            frames = self.pipe.vae.decode_stream_chunk(
                x0, output_device=torch.device("cpu"))
            torch.cuda.synchronize(self.device)
            decode_s = time.perf_counter() - decode_started
        if dist.is_initialized():
            dist.barrier()
        self.step_index += 1
        return frames, {
            "step": self.step_index,
            "camera_s": camera_s,
            "dit_s": dit_s,
            "decode_s": decode_s,
            "total_s": time.perf_counter() - started,
            "frames": 0 if frames is None else frames.shape[1],
        }

    def close(self):
        if not self.closed and self.rank == 0:
            self.pipe.vae.end_decode_stream()
        self.closed = True
        self.self_kv_cache = None
        self.cross_kv_cache = None
