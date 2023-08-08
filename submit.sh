#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -p llm
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --ntasks-per-node=4     # This needs to match Fabric(devices=...)
#SBATCH --gpus-per-task=1            # Request N GPUs per machine
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

# Run your training script
source /mnt/petrelfs/share_data/llm_env/env/llm-20230605
source /mnt/petrelfs/share_data/llm_env/env/llm-20230605

srun python -W ignore train.py --config ./configs/13B_sft.py 