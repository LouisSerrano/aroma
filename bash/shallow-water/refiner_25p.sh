#!/bin/bash

#SBATCH --partition=jazzy
#SBATCH --job-name=refiner
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate aroma 

dataset_name='shallow-water-dino'
same_grid=False
sub_from=1
sub_tr=0.0625
sub_te=0.0625
batch_size=64

#checkpoint_path=$WANDB_DIR/shallow-water-dino/inr/lilac-shadow-154.pt #worthy-field-152.pt
checkpoint_path=$WANDB_DIR/shallow-water-dino/inr/spring-vortex-281.pt #stilted-wind-271.pt

python3 dynamics_modeling/train_refiner.py data.dataset_name=$dataset_name data.same_grid=$same_grid data.sub_tr=$sub_tr data.sub_te=$sub_te wandb.checkpoint_path=$checkpoint_path optim.epochs=1000 
