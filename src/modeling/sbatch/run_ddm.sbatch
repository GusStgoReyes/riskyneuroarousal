#!/bin/bash
#SBATCH --job-name=pyddm
#SBATCH --time=06:00:00
#SBATCH -p russpold
#SBATCH --array=1-54

module load python/3.12.1
source /scratch/users/gustxsr/PoldrackLab/riskyneuroarousal/.venv/bin/activate

# Get subject ID from the subjects.csv file based on SLURM_ARRAY_TASK_ID
SUBJ_ID=$(awk -v task_id=$SLURM_ARRAY_TASK_ID 'NR == task_id + 1 {print $1}' /scratch/users/gustxsr/PoldrackLab/riskyneuroarousal/src/subjects.csv)

# Run the Python script with the extracted subject ID
python /scratch/users/gustxsr/PoldrackLab/riskyneuroarousal/src/modeling/pyddm_model.py --subj_ID "$SUBJ_ID" --model "1" --results_pth "/scratch/users/gustxsr/PoldrackLab/riskyneuroarousal/results"
