#!/bin/bash
#SBATCH -A plgccbmc11-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/test_with_params/output/%a/logs.out
#SBATCH --array=0-1

source venv/bin/activate
module add tensorflow/2.8.0-fosscuda-2020b

START=$(date +%s.%N)

python -m scripts.test_with_params.train_and_test.py ${SLURM_ARRAY_TASK_ID}

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo Elapsed $DIFF seconds
