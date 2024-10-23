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

dataset_name='navier-stokes-dino'
same_grid=False
sub_from=1
sub_tr=0.015625
sub_te=0.015625
batch_size=64

python3 inr/arom_inr.py data.same_grid=$same_grid data.sub_tr=$sub_tr data.sub_te=$sub_te inr.encode_geo=True
