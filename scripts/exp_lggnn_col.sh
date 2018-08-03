#!/bin/bash
#
#SBATCH --job-name=lgGnnCol
#SBATCH --output=slurm_out/lggnnCol.out
#SBATCH --error=slurm_out/lggnnCol.err
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --mem=250000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ds5821@nyu.edu
#SBATCH --constraint=gpu_12gb

bs=30
lr=0.0004
lrd=0.9
step=20
up=2
tr_path=/misc/vlgscratch4/BrunaGroup/sulem/chem/data/generated/cp_train_4800.pickle
te_path=/misc/vlgscratch4/BrunaGroup/sulem/chem/data/generated/cp_valid_600.pickle

python3 scripts/main_generate.py --lg True --update $up --bs $bs --lr $lr --lrdamping $lrd --step $step --train_path $tr_path --valid_path $te_path

echo done
