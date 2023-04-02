#!/bin/bash
#SBATCH -A plgccbmc11-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/model_selection/output/logs_convnet.out

module add tensorflow/2.8.0-fosscuda-2020b
source venv/bin/activate

START=$(date +%s.%N)

python -m scripts.model_selection.convnet.py

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo Elapsed $DIFF seconds