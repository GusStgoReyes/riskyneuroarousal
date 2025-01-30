#!/bin/bash
#SBATCH --job-name=hddm
#SBATCH --time=00:30:00
#SBATCH -p russpold
#SBATCH --array=2-54

# Get subject ID from the subjects.csv file based on SLURM_ARRAY_TASK_ID
SUBJ_ID=$(awk -v task_id=$SLURM_ARRAY_TASK_ID 'NR == task_id + 1 {print $1}' /scratch/users/gustxsr/PoldrackLab/riskyneuroarousal/src/subjects_non_pupil.csv)

# Run the Python script with the extracted subject ID from the HDDM container
singularity exec /scratch/users/gustxsr/PoldrackLab/hddm_latest.sif python /scratch/users/gustxsr/PoldrackLab/riskyneuroarousal/src/modeling/hddm_model.py --subj_ID "$SUBJ_ID" --model "1" --results_pth "/scratch/users/gustxsr/PoldrackLab/riskyneuroarousal/results" --model_pth "/scratch/users/gustxsr/PoldrackLab/riskyneuroarousal/results/models"


