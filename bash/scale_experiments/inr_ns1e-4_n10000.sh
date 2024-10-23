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

python3 inr/arom_inr.py --config-name inr_ns1e-4.yaml data.dir=/data/serrano/fno data.ntrain=9800 optim.epochs=100 inr.num_self_attentions=3 inr.num_latents=64 inr.max_pos_encoding_freq=5 inr.kl_weight=1e-5 inr.mlp_feature_dim=16 inr.dim=128 inr.encode_geo=True inr.include_pos_in_value=False optim.lr_inr=2e-3 inr.bottleneck_index=3
