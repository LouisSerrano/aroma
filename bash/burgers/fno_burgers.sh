#!/bin/bash

#SBATCH --partition=funky
#SBATCH --job-name=fno-burgers
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

python3 baselines/train.py --config-name fno_burgers.yaml #"data.ntrain=$ntrain" "data.ntest=$ntest" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dir=/data/wangt/mp_pde" "data.dataset_name=$dataset_name" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr_inr=$lr_inr" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "wandb.saved_checkpoint=$saved_checkpoint" 
