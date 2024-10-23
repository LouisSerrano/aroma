#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=refiner
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate aroma 

#dandy-voice-118
denim-snowball-112.pt

checkpoint_path=/data/serrano/functa2functa/navier-stokes-1e-4/inr/ethereal-bush-739.pt
python3 dynamics_modeling/train_refiner.py --config-name transformer_ns1e-4.yaml data.dir=/data/serrano/fno/ data.ntrain=9800 wandb.checkpoint_path=$checkpoint_path optim.epochs=100 optim.batch_size=32
