#!/bin/bash

#SBATCH --partition=funky
#SBATCH --job-name=inr
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate aroma 

dataset_name='navier-stokes-dino'
same_grid=False
sub_from=1
sub_tr=0.015625
sub_te=0.015625
batch_size=64
checkpoint_path=$WANDB_DIR/navier-stokes-dino/inr/quiet-rain-282.pt #happy-feather-160.pt #silvery-durian-81.pt

python3 dynamics_modeling/train_refiner.py data.same_grid=$same_grid data.sub_tr=$sub_tr data.sub_te=$sub_te wandb.checkpoint_path=$checkpoint_path optim.epochs=1000
