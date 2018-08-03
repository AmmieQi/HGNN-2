#!/bin/bash
#
#SBATCH --job-name=Test
#SBATCH --output=slurm_out/test.out
#SBATCH --error=slurm_out/test.err
#SBATCH --time=40:00:00
#SBATCH --nodes=8
#SBATCH --mem=15000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ds5821@nyu.edu
#SBATCH --constraint=gpu_12gb

lr='0.01 0.003 0.001 0.0003 0.0001'

for l in $lr
do

python3 scripts/main_gnn.py --bs 32 --lr $l --lrdamping 0.8 --h 5

done
echo All done
