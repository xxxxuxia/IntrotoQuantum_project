#!/bin/bash
# E1 depth sweep on a Slurm GPU cluster: one array task = one problem size = one GPU.
# Each task writes to its own results/E1_gpu/n<size>/ so parallel tasks never
# append to the same CSV. Resubmitting skips runs that already finished.
#
#   sbatch cluster/E1_gpu.sbatch                      # sizes from SIZES below
#   sbatch --array=0 cluster/E1_gpu.sbatch            # just the first size
#
# Edit the #SBATCH lines (partition, account, time) for your cluster.

#SBATCH --job-name=E1_depth
#SBATCH --partition=gpu               # TODO: your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/E1_%A_%a.out

SIZES=(14 16 18 20)
N=${SIZES[$SLURM_ARRAY_TASK_ID]}

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# One-time setup on a login node:
#   module load python/3.11 cuda/12      # names differ per cluster: `module avail`
#   python -m venv .venv && .venv/bin/pip install -r requirements.txt
# module load python/3.11 cuda/12
source .venv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv
python -u experiments/E1_depth.py --sizes "$N" --target nvidia --out-dir "results/E1_gpu/n$N"
