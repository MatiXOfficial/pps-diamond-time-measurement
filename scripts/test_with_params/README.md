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

### Run as a batch job
```bash
sbatch scripts/test_with_params/executor.sh
```
