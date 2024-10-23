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
sub_tr=0.0125
sub_te=0.0125
batch_size=16

checkpoint_path=$WANDB_DIR/shallow-water-dino/inr/pretty-leaf-290.pt #curious-bush-153.pt

python3 dynamics_modeling/train_refiner.py data.dataset_name=$dataset_name data.same_grid=$same_grid data.sub_tr=$sub_tr data.sub_te=$sub_te  wandb.checkpoint_path=$checkpoint_path optim.lr=1e-4 optim.min_noise=1e-3 optim.epochs=1000



