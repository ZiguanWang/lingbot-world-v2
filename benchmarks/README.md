# Benchmark artifact layout

Benchmark artifacts are grouped by GPU platform first, then by GPU count and VAE convolution implementation:

```text
benchmarks/
├── r9700/
│   ├── 4gpu_native3d/
│   ├── 4gpu_split2d/
│   ├── 8gpu_native3d/
│   ├── 8gpu_split2d/
│   ├── staged_sync_async/{logs,videos,profiles,conditions,latents}/
│   ├── ablations/async33/
│   └── vae17/
└── mi308/
    ├── 4gpu_native3d/
    ├── 4gpu_split2d/
    ├── 8gpu_native3d/
    ├── 8gpu_split2d/
    ├── 8gpu_split2d_chunk1/
    └── 4gpu_examples_257/
```

Naming rules:

- Hardware is always the first directory level (`r9700` or `mi308`).
- The main matrix name is `{gpu_count}_{native3d|split2d}`.
- R9700 log basenames start with `r9700_`, so copied logs remain identifiable outside this tree.
- Each complete-pipeline matrix keeps its own `logs/`, `videos/`, and `profiles/` when present.
- `staged_sync_async` is reserved for condition → DiT → standalone decode experiments; it is not the same timing
  boundary as a complete in-process pipeline.
- `ablations` and `vae17` contain diagnostic experiments, not the primary matrix.

Scripts under `scripts/` use these paths as their default output locations. Large traces, videos, latent tensors,
and downloaded weights should not be committed unless they are intentionally part of a published benchmark record.
Full local runs can be kept under `benchmarks/<platform>/local_runs/`. That subtree is ignored by Git; copy only the
small, curated summaries needed for publication into the corresponding tracked configuration directory.
