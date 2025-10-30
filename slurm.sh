#!/bin/bash
#SBATCH --job-name=chatSkibidi_train
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                    #can change these based on needs
#SBATCH --cpus-per-task=16            #can change these based on needs
#SBATCH --mem=32G                            #can change these based on needs
#SBATCH --requeue                       # allow restart if preempted
#SBATCH --output=logs/%j.out  # %j: job ID, %x: job name. Reference: https://slurm.schedmd.com/sbatch.html#lbAH
 
#Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo ""

#Activate virtual environment if you have one
#source /path/to/your/venv/bin/activate

#Run your Python script
echo "Running llm.py..."
python3 llm.py

#Print completion information
echo ""
echo "Job finished at: $(date)"
