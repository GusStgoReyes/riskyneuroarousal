#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --time=00:30:00
#SBATCH -p russpold,owners,normal
#SBATCH --array=0-499

source /scratch/users/gustxsr/riskyneuroarousal/.venv/bin/activate

python /scratch/users/gustxsr/riskyneuroarousal/src/riskyneuroarousal/modeling/run_simulations.py 7 $SLURM_ARRAY_TASK_ID 500 0 /scratch/users/gustxsr/results_sim
