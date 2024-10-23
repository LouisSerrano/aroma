#!/bin/bash

#SBATCH --partition=jazzy
#SBATCH --job-name=airfoil
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --time=5000

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate aroma 

python3 dynamics_modeling/train_refiner_graph.py --config-name transformer_airfoil.yaml optim.epochs=1000 optim.min_noise=1e-6
