#!/bin/bash
#SBATCH --job-name=fitting
#SBATCH --time=01:00:00
#SBATCH -p normal,russpold
#SBATCH --array=0-53

source /scratch/users/gustxsr/riskyneuroarousal/.venv/bin/activate

# Limit thread usage for numpy/scipy
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python /scratch/users/gustxsr/riskyneuroarousal/src/riskyneuroarousal/modeling/fit_participants.py 2 $SLURM_ARRAY_TASK_ID 54 /scratch/users/gustxsr/results_fit /scratch/users/gustxsr/riskyneuroarousal/config.json
