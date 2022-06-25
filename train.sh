#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=10Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module load anaconda/3
module load cuda/11.1
conda activate stylegan3

python train.py \
--outdir=<output_path> \
--cfg=stylegan2 \
--data=<path_to_generated_mel_spectrograms> \
--gpus=1 \
--batch=32 \
--aug=noaug \
--gamma=0.0512 \
--cond=True