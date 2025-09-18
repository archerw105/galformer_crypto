#!/bin/bash
#SBATCH --job-name=galformer_pytorch
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=04:00:00
#SBATCH --output=galformer_%j.out
#SBATCH --error=galformer_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=archerdwang@gmail.com

# Load required modules
module load  python/miniforge3_pytorch/2.5.0
module load cuda/12.3

# Activate conda environment
source galformer/bin/activate

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Change to working directory
cd /u/archerdw/galformer_project

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"

# Run the Galformer PyTorch model
python Galformer_pytorch_fixed.py

echo "End Time: $(date)"
echo "Job completed successfully"
