#!/bin/bash
#
#SBATCH --job-name=gnn011
#SBATCH --output=slurm_out/gnn_011.out
#SBATCH --error=slurm_out/gnn_011.err
#SBATCH --time=40:00:00
#SBATCH --nodes=4
#SBATCH --mem=250000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ds5821@nyu.edu
#SBATCH --constraint=gpu_12gb

lr=0.0003
lrdamp=0.9
step=10
epochs=40
L=15
data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_0.pickle'
h=1

python3 scripts/main_gnn_qm9_2.py --lr $lr --lrdamping $lrdamp --step $step --data_path $data_path --epochs $epochs --L $L --h $h

echo done

