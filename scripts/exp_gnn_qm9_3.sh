#!/bin/bash
#
#SBATCH --job-name=gnn33
#SBATCH --output=slurm_out/gnn_33.out
#SBATCH --error=slurm_out/gnn_33.err
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --mem=250000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ds5821@nyu.edu
#SBATCH --constraint=gpu_12gb

lr=0.0004
lrdamp=0.9
step=5
epochs=20
data_path='/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_3.pickle'

python3 scripts/main_gnn_qm9.py --lr $lr --lrdamping $lrdamp --step $step --data_path $data_path --epochs $epochs

echo done

