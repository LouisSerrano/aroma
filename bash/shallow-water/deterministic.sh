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
sub_tr=2
sub_te=2
batch_size=64

#checkpoint_path=$WANDB_DIR/shallow-water-dino/inr/worthy-field-152.pt
checkpoint_path=$WANDB_DIR/shallow-water-dino/inr/decent-surf-288.pt

python3 dynamics_modeling/ablations/train_deterministic.py data.dataset_name=$dataset_name data.same_grid=$same_grid data.sub_tr=$sub_tr data.sub_te=$sub_te wandb.checkpoint_path=$checkpoint_path optim.epochs=1000 optim.min_noise=1e-3 optim.batch_size=16 optim.threshold=2.0 optim.epochs=2000 
