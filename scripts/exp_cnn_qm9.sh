#!/bin/bash
#
#SBATCH --job-name=ccn1_0
#SBATCH --output=slurm_out/ccn1_0.out
#SBATCH --error=slurm_out/ccn1_0.err
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --mem=250000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ds5821@nyu.edu
#SBATCH --constraint=gpu_12gb

lr=0.001
lrd=0.9
step=10
data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_0.pickle'
epochs=30

python3 scripts/main_ccn_qm9.py --lr $lr --lrdamping $lrd --step $step --epochs $epochs --data_path $data_path

echo done
