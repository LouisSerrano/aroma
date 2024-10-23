#!/bin/bash

#SBATCH --partition=jazzy
#SBATCH --job-name=inr
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate aroma 


python3 dynamics_modeling/ablations/train_mlp.py --config-name transformer_burgers.yaml 
