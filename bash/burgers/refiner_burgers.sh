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

dataset_name='mp-pde-burgers'
model_type='siren'
same_grid=True
sub_from=1
sub_tr=2
sub_te=2
seq_inter_len=20
seq_extra_len=5
batch_size=128
lr_inr=1e-4
epochs=10000
latent_dim=64
depth=4
hidden_dim=128
saved_checkpoint=False
ntrain=2048
ntest=128

python3 dynamics_modeling/train_refiner.py --config-name transformer_burgers.yaml 
