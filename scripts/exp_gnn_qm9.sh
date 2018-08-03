#!/bin/bash
#
#SBATCH --job-name=gnn_12
#SBATCH --output=slurm_out/gnn_qm9_12.out
#SBATCH --error=slurm_out/gnn_qm9_12.err
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --mem=250000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ds5821@nyu.edu
#SBATCH --constraint=gpu_12gb

lr=0.0003
lrdamp=0.9
step=5
epochs=20
L=15
bs=30
data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9.pickle'

python3 scripts/main_gnn_qm9_2.py --lr $lr --lrdamping $lrdamp --step $step --data_path $data_path --epochs $epochs --bs $bs --L $L

echo done

