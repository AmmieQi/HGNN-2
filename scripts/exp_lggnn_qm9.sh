#!/bin/bash
#
#SBATCH --job-name=lgGnn_10
#SBATCH --output=slurm_out/lggnn_qm9_10.out
#SBATCH --error=slurm_out/lggnn_qm9_10.err
#SBATCH --time=40:00:00
#SBATCH --nodes=4
#SBATCH --mem=250000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ds5821@nyu.edu
#SBATCH --constraint=gpu_12gb

lr=0.0004
lrd=0.9
step=10
epochs=20
L=5
bs=30
data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9.pickle'
u=2
h=1

python3 scripts/main_gnn_qm9_2.py --lg True --update $u --lr $lr --lrdamping $lrd --step $step --epochs $epochs --bs $bs --data_path $data_path --L $L --h $h --shuffle True

echo done
