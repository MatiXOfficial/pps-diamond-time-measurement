### Prepare dataset
Download `data/dataset.npz`

### Prepare env:
```bash
module add tensorflow/2.8.0-fosscuda-2020b
python -m venv venv
source venv/bin/activate
pip install pandas scikit_learn matplotlib statsmodels
```

### Run
```
srun -A plgccbmc11-gpu -p plgrid-gpu-v100 --time=0:10:00 --gres=gpu:1 --pty /bin/bash -l
source venv/bin/activate
module add tensorflow/2.8.0-fosscuda-2020b
python -m scripts.test_train.py
```