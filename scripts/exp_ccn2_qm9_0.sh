#!/bin/bash
#
#SBATCH --job-name=ccn2_03
#SBATCH --output=slurm_out/ccn2_03.out
#SBATCH --error=slurm_out/ccn2_03.err
#SBATCH --time=40:00:00
#SBATCH --nodes=5
#SBATCH --mem=250000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ds5821@nyu.edu
#SBATCH --constraint=gpu_12gb

lr=0.0004
lrd=0.9
step=10
data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_0.pickle'
epochs=40
L=3

python3 scripts/main_ccn_qm9.py --k 2 --lr $lr --lrdamping $lrd --step $step --epochs $epochs --data_path $data_path --L $L

echo done
