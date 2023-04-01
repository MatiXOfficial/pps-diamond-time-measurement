#!/bin/bash
#SBATCH -A plgccbmc11-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --time=0:01:00
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/unet_tuning_ch30/output/%a/logs.out
#SBATCH --array=0-23

source venv/bin/activate
module add tensorflow/2.8.0-fosscuda-2020b

START=$(date +%s.%N)

python -m scripts.unet_tuning_ch30.train_and_test.py ${SLURM_ARRAY_TASK_ID}

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo Elapsed $DIFF seconds
