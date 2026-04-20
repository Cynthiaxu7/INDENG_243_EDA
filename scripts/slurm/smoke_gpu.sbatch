#!/bin/bash
#SBATCH --job-name=synth_smoke
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/results/slurm-%j.out
#SBATCH --error=artifacts/results/slurm-%j.err

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate fm4npp

set -euo pipefail

cd /home/shuhang/YBJ/mamba_stuff/INDENG_243_EDA
mkdir -p artifacts/results

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "hostname : $(hostname)"
echo "date     : $(date -Iseconds)"
nvidia-smi -L
python -c "import torch; print('cuda device:', torch.cuda.get_device_name(0))"

python scripts/train_synthetic.py \
    --device cuda \
    --models mamba,mamba_ssm_v1 \
    --noise-levels low \
    --mamba-epochs 3 \
    --mamba-d-model 32 \
    --mamba-n-blocks 1 \
    --mamba-d-state 8 \
    --nn-epochs 3 \
    --results-suffix _smoke
