#!/usr/bin/env bash
#SBATCH --job-name=hm3d-v2-convert-0919
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=176
#SBATCH --time=08:00:00
#SBATCH --output=/data/glx/Enactive/navigation/datasets/train/hm3d-v2/convert_%j.out
#SBATCH --error=/data/glx/Enactive/navigation/datasets/train/hm3d-v2/convert_%j.err

set -euo pipefail

repo_root="/workspace/glx/projects/xNav-data-preprocess"
input_root="/data/glx/Enactive/navigation/datasets/train/hm3d-v2/raw/hm3d_v2_0919_40k"
output_root="/data/glx/Enactive/navigation/datasets/train/hm3d-v2/processed/hm3d_v2_0919_40k"
workers="${HM3D_V2_WORKERS:-176}"

if [[ "${SLURM_CPUS_PER_TASK:-0}" -lt "${workers}" ]]; then
  echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset} is below workers=${workers}" >&2
  exit 2
fi

mkdir -p "$(dirname "${output_root}")"
cd "${repo_root}"
exec "${repo_root}/.venv/bin/python" "${repo_root}/ovon.py" \
  --input-root "${input_root}" \
  --output-root "${output_root}" \
  --dataset-name hm3d_v2 \
  --num-workers "${workers}"
