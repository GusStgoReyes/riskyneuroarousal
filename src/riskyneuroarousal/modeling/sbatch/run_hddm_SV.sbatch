#!/bin/bash
#SBATCH --job-name=hddm
#SBATCH --time=00:30:00
#SBATCH -p russpold,owners,normal
#SBATCH --array=1-1

# Get subject ID from the subjects.csv file based on SLURM_ARRAY_TASK_ID
SUBJ_ID=$(awk -v task_id=$SLURM_ARRAY_TASK_ID 'NR == task_id + 1 {print $1}' /scratch/users/gustxsr/riskyneuroarousal/src/riskyneuroarousal/subjects.csv)
MODEL_ID=1
# Run the Python script with the extracted subject ID from the HDDM container
singularity exec /home/groups/russpold/hddm_container/hddm_latest.sif python /scratch/users/gustxsr/riskyneuroarousal/src/riskyneuroarousal/modeling/hddm_model_SVs.py --subj_ID "$SUBJ_ID" --model "$MODEL_ID" --results_pth "/scratch/users/gustxsr/results_ddm/parameters" --model_pth "/scratch/users/gustxsr/results_ddm/trained_models"


