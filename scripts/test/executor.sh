#!/bin/bash
#SBATCH -A plgccbmc11-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/test/output/logs.out

source venv/bin/activate
module add tensorflow/2.8.0-fosscuda-2020b

START=$(date +%s.%N)

python -m scripts.test.train_and_test.py

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo Elapsed $DIFF seconds
