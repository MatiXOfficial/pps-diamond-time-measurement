import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.cross_validator import KerasTunerCrossValidator

tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import keras_tuner as kt

from src.models import mlp_builder as bare_model_builder
from src.utils import augmentation_random_cut

########## Setup ##########
print('Setup...')
PWD = '.'

CHANNEL = 17
N_BASELINE = 20

PROJECT_NAME = 'mlp'
DATASET_PATH = PWD + '/data/dataset/dataset.pkl'
TRIALS_DIR = PWD + f'/data/model_selection/channel_{CHANNEL}/tuner'
CROSSVAL_DIR = PWD + f'/data/model_selection/channel_{CHANNEL}/cross_val'

LR = 0.01

N_EPOCHS = 3000
BATCH_SIZE = 2048
MAX_TRIALS = 4
EXECUTIONS_PER_TRIAL = 1

TOP_N = 2
CROSSVAL_N_CV = 3
CROSSVAL_N_EXEC = 1
LOSS_WEIGHT = 1000

########## Load data ##########
print('Loading dataset...')
dataset = np.load(PWD + f'/data/dataset.npz', allow_pickle=True)

all_channels_data = dataset['dataset'].flat[0]
all_channels_data.keys()

all_X, all_y = all_channels_data[CHANNEL][0], all_channels_data[CHANNEL][1]

print(all_X.shape, all_y.shape)

########## Preprocess ##########
print('Preprocessing...')

all_X -= np.mean(all_X[:, :N_BASELINE], axis=1)[:, None]
all_X /= all_X.max(axis=1)[:, None]

X_aug, y_aug = augmentation_random_cut(all_X, all_y, 8, seed=42, apply=True)

X_train_default, _, y_train_default, _ = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_default, y_train_default, test_size=0.2, random_state=42)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

########## Model ##########
print('Model...')


def model_builder(hp: kt.HyperParameters) -> keras.Model:
    hp_n_hidden_layers = hp.Int("n_hidden_layers", min_value=1, max_value=6, step=1, default=2)
    hp_units_mult = hp.Choice("units_mult", values=[1, 2, 4, 8, 16, 32], default=4)
    hp_batch_normalization = hp.Boolean("batch_normalization", default=False)
    hp_input_batch_normalization = hp.Boolean("input_batch_normalization", default=False)
    hp_dropout = hp.Choice("dropout", values=[0.0, 0.2, 0.5])

    model = bare_model_builder(hp_n_hidden_layers, hp_units_mult, hp_batch_normalization, hp_input_batch_normalization,
                               hp_dropout)
    model.compile(loss='mse', optimizer=optimizers.Adam(LR), loss_weights=LOSS_WEIGHT)
    return model


print('Example summary')
print(model_builder(kt.HyperParameters()).summary())

model_callbacks = [
    callbacks.EarlyStopping(patience=50),
    callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10)
]

########## Tuning ##########
print('Tuning...')

bayesian_tuner = kt.BayesianOptimization(model_builder, objective='val_loss', executions_per_trial=EXECUTIONS_PER_TRIAL,
                                         max_trials=MAX_TRIALS, directory=TRIALS_DIR, project_name=PROJECT_NAME,
                                         overwrite=False)

bayesian_tuner.search(X_train, y_train, validation_data=[X_val, y_val], epochs=N_EPOCHS, callbacks=model_callbacks,
                      batch_size=BATCH_SIZE, verbose=0)

print('Best models')
for i, hyperparameters in enumerate(bayesian_tuner.get_best_hyperparameters(TOP_N)):
    print(f'========== Model {i} ==========')
    print(hyperparameters.get_config()['values'])
    model_tmp = model_builder(hyperparameters)
    print('Number of parameters:', model_tmp.count_params())

########## Cross-validation ##########
print('Cross-validation...')

cross_validator = KerasTunerCrossValidator(bayesian_tuner, X_train_default, y_train_default, model_builder,
                                           directory=CROSSVAL_DIR, project_name=PROJECT_NAME,
                                           n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, n_top=TOP_N,
                                           n_cv=CROSSVAL_N_CV, n_executions=CROSSVAL_N_EXEC, overwrite=False)
model_scores = cross_validator()

mean_scores = [f"{np.mean(scores):0.2f}" for scores in model_scores.values()]
std_scores = [f"{np.std(scores):0.2f}" for scores in model_scores.values()]
n_params = [model_builder(hyperparameters).count_params() for hyperparameters in
            bayesian_tuner.get_best_hyperparameters(TOP_N)]

df = pd.DataFrame({'mean': mean_scores, 'std': std_scores, 'n_params': n_params}, index=list(model_scores.keys()))
df.index.name = 'Model'

print('Final results:')
print(df)
