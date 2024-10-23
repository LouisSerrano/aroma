#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=inr
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate aroma 

checkpoint_path=$WANDB_DIR/navier-stokes-1e-4/inr/bumbling-sun-111.pt
python3 dynamics_modeling/train_refiner.py --config-name transformer_ns1e-4.yaml data.ntrain=1000 wandb.checkpoint_path=$checkpoint_path
