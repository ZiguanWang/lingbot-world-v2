#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH-}"

actual_frames() {
    case "$1" in
        17) echo 13 ;;
        33) echo 29 ;;
        65) echo 61 ;;
        129) echo 125 ;;
        257) echo 253 ;;
        *) return 2 ;;
    esac
}

mkdir -p benchmarks/r9700/staged_sync_async/logs benchmarks/r9700/staged_sync_async/videos
summary="benchmarks/r9700/staged_sync_async/logs/r9700_full_staged_matrix_summary.log"
: > "$summary"

for profile_mode in normal profile; do
    for mode in sync async; do
        for request in 17 33 65 129 257; do
            actual=$(actual_frames "$request")
            suffix=""
            extra=()
            if [[ "$profile_mode" == profile ]]; then
                suffix="_profile"
                extra=(profile)
            fi
            printf 'MATRIX_START profile=%s mode=%s request=%s actual=%s utc=%s\n' \
                "$profile_mode" "$mode" "$request" "$actual" "$(date -u +%FT%TZ)" | tee -a "$summary"
            ./scripts/r9700/run_staged_video.sh "$mode" "$request" "${extra[@]}"
            video="benchmarks/r9700/staged_sync_async/videos/${mode}_${request}req_${actual}actual${suffix}.mp4"
            python scripts/common/verify_generated_video.py "$video" --expected-frames "$actual" | tee -a "$summary"
            printf 'MATRIX_DONE profile=%s mode=%s request=%s actual=%s utc=%s\n' \
                "$profile_mode" "$mode" "$request" "$actual" "$(date -u +%FT%TZ)" | tee -a "$summary"
        done
    done
done

printf 'MATRIX_COMPLETE utc=%s\n' "$(date -u +%FT%TZ)" | tee -a "$summary"
