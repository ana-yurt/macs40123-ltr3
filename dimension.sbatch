#!/bin/bash
#SBATCH --job-name=dimension
#SBATCH --output=dimension%A_%a.log
#SBATCH --error=dimension%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --account=macs40123
#SBATCH --partition=caslake
#SBATCH --mem=30G
#SBATCH --array=0
#SBATCH --time=05:00:00 

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

# Load necessary modules and activate environment
module load python
eval "$(conda shell.bash hook)"
conda activate myenv

python dimension.py \
  --output_dir results_counterfactual \
  --model_dir embeddings_5_counterfactual \
  --start_year 1930 \
  --end_year 2005 \
  --max_workers 20