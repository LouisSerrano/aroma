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

dataset_name='navier-stokes-dino'
same_grid=True
sub_from=1
sub_tr=4
sub_te=4
batch_size=64

python3 inr/arom_inr.py data.same_grid=$same_grid data.sub_tr=$sub_tr data.sub_te=$sub_te inr.num_latents=32 inr.encode_geo=True 
