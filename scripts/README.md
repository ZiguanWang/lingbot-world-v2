# Script layout

The public model entry point is now `scripts/common/generate.py`. Operational, benchmark, and diagnostic scripts are
grouped by platform:

```text
scripts/
├── common/
│   ├── generate.py
│   ├── run_fast.sh
│   ├── run_rank_numa_bound.sh
│   ├── check_checkpoint_values.py
│   ├── compare_latents.py
│   ├── gpu_health_check.py
│   ├── interactive_server.py
│   ├── verify_interactive_mvp.py
│   └── verify_generated_video.py
├── r9700/
│   ├── run_{4gpu,8gpu}_*.sh
│   ├── run_async*.sh
│   ├── run_staged_video.sh
│   ├── run_full_staged_matrix.sh
│   ├── run_vae_17_*.sh
│   ├── dit_only_memory_probe.py
│   └── vae_memory_probe.py
└── mi308/
    ├── run_mi308_4gpu.sh
    ├── run_mi308_4gpu_matrix.sh
    └── run_mi308_interactive_4gpu.sh
```

All shell scripts resolve their own location, switch to the repository root, and add that root to `PYTHONPATH`.
They can therefore be launched from outside the repository without changing their output paths. Examples:

```bash
./scripts/common/run_fast.sh lingbot-world-v2-14b-causal-fast 361
./scripts/r9700/run_8gpu_async_split2d.sh 257
./scripts/mi308/run_mi308_4gpu_matrix.sh
./scripts/mi308/run_mi308_interactive_4gpu.sh
```

Machine-specific scripts write to the matching subtree under `benchmarks/`. Empty platform directories are
intentionally not created.
