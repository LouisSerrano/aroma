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

dataset_name='shallow-water-dino'
same_grid=False
sub_from=1
sub_tr=0.0625
sub_te=0.0625
batch_size=64

python3 inr/arom_inr.py --config-name inr_shallow_water.yaml data.dataset_name=$dataset_name data.same_grid=$same_grid data.sub_tr=$sub_tr data.sub_te=$sub_te inr.encode_geo=True inr.dropout_sequence=0.5 inr.num_self_attentions=3
