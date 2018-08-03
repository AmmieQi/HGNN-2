#!/bin/bash
#
#SBATCH --job-name=gnn13
#SBATCH --output=slurm_out/gnn_13.out
#SBATCH --error=slurm_out/gnn_13.err
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --mem=250000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ds5821@nyu.edu
#SBATCH --constraint=gpu_12gb

lr=0.0004
lrdamp=0.9
step=10
epochs=25
data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_1.pickle'

python3 scripts/main_gnn_qm9.py --lr $lr --lrdamping $lrdamp --step $step --data_path $data_path --epochs $epochs

echo done

