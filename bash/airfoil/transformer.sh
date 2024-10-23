#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=airfoil
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate aroma 

python3 dynamics_modeling/arom_deterministic_graph.py --config-name transformer_airfoil.yaml optim.epochs=100
