#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
work_root=${HUMAN_REPLAY_WORK_ROOT:-/data/glx/Enactive/navigation/staging/human-replay-20260902-20260908}
run_id=${HUMAN_REPLAY_RUN_ID:-human-vln-20260902-20260908-v2-qwen37-10fps}
output_root=${HUMAN_REPLAY_OUTPUT_ROOT:-/data/glx/Enactive/navigation/datasets/train/DataEngine/human_replay_20260902_20260908/vln}
job_dir="$work_root/annotation/.runs/$run_id"

# Refresh the public JSON handoff using local state; no model/API calls.
"$repo_root/../navigation-process/.venv/bin/nav-process" report --job-dir "$job_dir"
PYTHONUNBUFFERED=1 "$repo_root/.venv/bin/python" "$repo_root/human_replay.py" \
  --input-root "$work_root" --annotation-report "$job_dir/report.json" \
  --output-root "$output_root" "$@"
