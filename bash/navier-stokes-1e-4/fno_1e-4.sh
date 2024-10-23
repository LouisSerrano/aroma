#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=fno-1e-4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate aroma 


python3 baselines/train_2d.py --config-name fno_1e-4.yaml #"data.ntrain=$ntrain" "data.ntest=$ntest" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dir=/data/wangt/mp_pde" "data.dataset_name=$dataset_name" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr_inr=$lr_inr" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "wandb.saved_checkpoint=$saved_checkpoint" 
