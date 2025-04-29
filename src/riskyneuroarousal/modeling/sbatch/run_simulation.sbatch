#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --time=01:00:00
#SBATCH -p russpold,owners,normal
#SBATCH --array=0-0

module load python/3.12.1
source /scratch/users/gustxsr/PoldrackLab/riskyneuroarousal/.venv/bin/activate

python /scratch/users/gustxsr/PoldrackLab/riskyneuroarousal/src/modeling/run_simulations.py 0 $SLURM_ARRAY_TASK_ID 1 0 /scratch/users/gustxsr/PoldrackLab/