#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --time=01:00:00
#SBATCH -p normal,russpold
#SBATCH --array=0-0

source /scratch/users/gustxsr/riskyneuroarousal/.venv/bin/activate

# Limit thread usage for numpy/scipy
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python /scratch/users/gustxsr/riskyneuroarousal/src/riskyneuroarousal/modeling/run_simulations.py 0 $SLURM_ARRAY_TASK_ID 1 0 /scratch/users/gustxsr/results_sim
