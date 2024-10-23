#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=cylinder
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate aroma 

python3 dynamics_modeling/train_refiner_graph.py --config-name transformer_cylinder.yaml optim.epochs=1000 optim.min_noise=1e-3 wandb.checkpoint_path="/data/serrano/functa2functa/cylinder-flow/inr/golden-silence-751.pt" data.dir="/data/serrano/"
