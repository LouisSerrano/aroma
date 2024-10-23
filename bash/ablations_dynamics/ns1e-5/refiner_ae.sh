#!/bin/bash

#SBATCH --partition=hard
#SBATCH --job-name=inr
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate aroma 

checkpoint_path=$WANDB_DIR/navier-stokes-1e-5/inr/graceful-gorge-539.pt
python3 dynamics_modeling/train_refiner.py --config-name transformer_ns1e-5.yaml wandb.checkpoint_path=$checkpoint_path
