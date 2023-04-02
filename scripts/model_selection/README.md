### Prepare dataset
Download `data/dataset.npz`

- Run everything from the root directory

### Prepare env:
```bash
module add tensorflow/2.8.0-fosscuda-2020b
python -m venv venv
source venv/bin/activate
pip install pandas scikit_learn matplotlib statsmodels
pip install --no-deps keras-tuner
```

### Run as a batch job
```bash
sbatch scripts/model_selection/mlp_executor.sh
sbatch scripts/model_selection/convnet_executor.sh
```

### Run interactively
```bash
srun -A plgccbmc11-gpu -p plgrid-gpu-v100 --time=0:30:00 --gres=gpu:1 --pty /bin/bash -l
module add tensorflow/2.8.0-fosscuda-2020b
source venv/bin/activate

python -m scripts.model_selection.mlp
python -m scripts.model_selection.convnet
```
