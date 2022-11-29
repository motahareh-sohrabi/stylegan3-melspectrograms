#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=10Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module load anaconda/3
module load cuda/11.1
conda activate stylegan3

python stylegan3/train.py \
--outdir="/network/scratch/m/motahareh.sohrabi/exvo/stylegan2_output$" \
--cfg=stylegan2 \
--data="/network/scratch/m/motahareh.sohrabi/exvo/clean_test" \
--gpus=1 \
--batch=32 \
--aug=noaug \
--gamma=0.0512 \
--cond=True