#!/bin/bash
#SBATCH --partition=gpu-ffa         # partition you want to run job in
#SBATCH --gpus=3
#SBATCH --time=1-00:00:00         # walltime for the job in format (days-)hours:minutes:seconds
#SBATCH --job-name="albert_1"     # change to your job name
#SBATCH --output=output_albert_1.txt       # stdout and stderr output file
#SBATCH --mail-user=hamza.gamouh@gmail.com # send email when job changes state to email address 

export LD_LIBRARY_PATH=/usr/local/cuda/lib64

srun ch-run biopython python ./app/prediction_bs_torch.py

