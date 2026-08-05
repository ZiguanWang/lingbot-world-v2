#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"

# This host has eight NUMA nodes.  Node N owns CPUs N*8..N*8+7 and
# N*8+64..N*8+71.  Keep each torchrun worker on one node so the full model
# load and FSDP flatten first-touch local memory instead of triggering slow
# automatic NUMA page migration across ranks.
rank=${LOCAL_RANK:?LOCAL_RANK is required}
low_start=$((rank * 8))
low_end=$((low_start + 7))
high_start=$((low_start + 64))
high_end=$((high_start + 7))

exec taskset -c "${low_start}-${low_end},${high_start}-${high_end}" \
    python -u scripts/common/generate.py "$@"
