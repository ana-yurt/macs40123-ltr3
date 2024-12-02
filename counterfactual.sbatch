#!/bin/bash
#SBATCH --job-name=counterfactual
#SBATCH --output=counterfactual%A_%a.log
#SBATCH --error=counterfactual%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --account=macs40123
#SBATCH --partition=caslake
#SBATCH --mem=10G
#SBATCH --array=0-19
#SBATCH --time=05:00:00 

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR
module load python
eval "$(conda shell.bash hook)"
conda activate myenv
SAMPLEID=$SLURM_ARRAY_TASK_ID

python counterfactual.py --sample SAMPLEID --workers 10
