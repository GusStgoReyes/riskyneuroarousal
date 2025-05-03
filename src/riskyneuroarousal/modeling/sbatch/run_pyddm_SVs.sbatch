#!/bin/bash
#SBATCH --job-name=pyddm
#SBATCH --time=06:00:00
#SBATCH -p russpold
#SBATCH --array=1-54

source /scratch/users/gustxsr/riskyneuroarousal/.venv/bin/activate

# Get subject ID from the subjects.csv file based on SLURM_ARRAY_TASK_ID
SUBJ_ID=$(awk -v task_id=$SLURM_ARRAY_TASK_ID 'NR == task_id + 1 {print $1}' /scratch/users/gustxsr/riskyneuroarousal/src/subjects.csv)

# Run the Python script with the extracted subject ID
python /scratch/users/gustxsr/riskyneuroarousal/src/modeling/pyddm_model_SVs.py --subj_ID "$SUBJ_ID" --results_pth "/scratch/users/gustxsr/DDM_results"
