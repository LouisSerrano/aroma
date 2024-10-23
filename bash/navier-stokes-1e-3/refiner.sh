#!/bin/bash

#SBATCH --partition=funky
#SBATCH --job-name=refiner
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate aroma 

dataset_name='navier-stokes-dino'
same_grid=True
sub_from=1
sub_tr=4
sub_te=4
batch_size=16
#checkpoint_path=$WANDB_DIR/navier-stokes-dino/inr/royal-shadow-159.pt 
checkpoint_path=$WANDB_DIR/navier-stokes-dino/inr/sunny-elevator-287.pt

python3 dynamics_modeling/train_refiner.py data.same_grid=$same_grid data.sub_tr=$sub_tr data.sub_te=$sub_te wandb.checkpoint_path=$checkpoint_path optim.epochs=5000 optim.min_noise=1e-3
