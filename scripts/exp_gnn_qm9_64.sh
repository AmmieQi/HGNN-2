#!/bin/bash
#
#SBATCH --job-name=gnn64gb
#SBATCH --output=slurm_out/gnn64gb.out
#SBATCH --error=slurm_out/gnn64gb.err
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --mem=64000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ds5821@nyu.edu
#SBATCH --constraint=gpu_12gb

bs=30
lr=0.01
lrdamp=0.8
step=10
tr_path=/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/train_10000.pickle
te_path=/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/test_1000.pickle

python3 scripts/main_gnn.py --bs $bs --lr $lr --lrdamping $lrdamp --step $step --train_path $tr_path --test_path $te_path

echo done

