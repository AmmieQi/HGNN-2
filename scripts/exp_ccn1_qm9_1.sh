#!/bin/bash
#
#SBATCH --job-name=ccn1_11
#SBATCH --output=slurm_out/ccn1_11.out
#SBATCH --error=slurm_out/ccn1_11.err
#SBATCH --time=40:00:00
#SBATCH --nodes=5
#SBATCH --mem=250000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ds5821@nyu.edu
#SBATCH --constraint=gpu_12gb

lr=0.0004
lrd=0.9
step=10
data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_1.pickle'
epochs=25
L=6

python3 scripts/main_ccn_qm9.py --lr $lr --lrdamping $lrd --step $step --epochs $epochs --data_path $data_path --L $L

echo done
