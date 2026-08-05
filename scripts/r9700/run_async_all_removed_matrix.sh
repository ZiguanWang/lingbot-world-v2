#!/bin/bash
set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"

mkdir -p benchmarks/r9700/8gpu_native3d/logs
summary="benchmarks/r9700/8gpu_native3d/r9700_summary.log"
: > "$summary"
for frames in 33 65 129 257; do
    printf 'START frames=%s utc=%s\n' "$frames" "$(date -u +%FT%TZ)" | tee -a "$summary"
    if ./scripts/r9700/run_async_all_removed.sh "$frames"; then
        rc=0
    else
        rc=$?
    fi
    printf 'DONE frames=%s rc=%s utc=%s\n' "$frames" "$rc" "$(date -u +%FT%TZ)" | tee -a "$summary"
    python scripts/common/gpu_health_check.py 2>&1 | tee -a "$summary"
done
printf 'MATRIX_COMPLETE utc=%s\n' "$(date -u +%FT%TZ)" | tee -a "$summary"
