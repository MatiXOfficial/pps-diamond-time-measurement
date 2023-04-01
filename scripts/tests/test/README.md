### Prepare dataset
Download `data/dataset.npz`

- Run everything from the root directory
- Note the `train` parameter in `train_model`

### Prepare env:
```bash
module add tensorflow/2.8.0-fosscuda-2020b
python -m venv venv
source venv/bin/activate
pip install pandas scikit_learn matplotlib statsmodels
```

### Run interactively
```bash
srun -A plgccbmc11-gpu -p plgrid-gpu-v100 --time=0:10:00 --gres=gpu:1 --pty /bin/bash -l
source venv/bin/activate
module add tensorflow/2.8.0-fosscuda-2020b
python -m scripts.test.train_and_test.py
```

### Run as a batch job
```bash
sbatch scripts/test/executor.sh
```
