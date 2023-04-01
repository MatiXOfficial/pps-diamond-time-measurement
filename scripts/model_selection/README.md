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
```
