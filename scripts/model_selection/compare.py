import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.get_logger().setLevel('ERROR')
from tensorflow.keras import optimizers

from src.models import optimal_model_builders
from src.utils import augmentation_random_cut
from src.cfd_utils import plot_difference_hist, _get_gauss_stats, TIME_STEP
from src.cross_validator import CrossValidator

########## Setup ##########
print('Setup...')
PWD = '.'
if os.getenv('SCRATCH'):
    PWD_TMP = os.getenv('SCRATCH') + '/pps-diamond-time-measurement'
else:
    PWD_TMP = '.'

print('PWD:', PWD)
print('PWD_TMP:', PWD_TMP)

CHANNEL = 17
N_BASELINE = 20

OVERWRITE = False

PROJECT_NAME = 'compare'
TRIALS_DIR = PWD_TMP + f'/data/model_selection/channel_{CHANNEL}/tuner'
CROSSVAL_DIR = PWD_TMP + f'/data/model_selection/channel_{CHANNEL}/cross_val'

LR = 0.01
ES_MIN_DELTA_REGULAR = 0.1
ES_MIN_DELTA_HEATMAP = 0.01

N_EPOCHS = 3000
BATCH_SIZE = 2048

CROSSVAL_N_CV = 5
CROSSVAL_N_EXEC = 3
LOSS_WEIGHT_REGULAR = 1000
LOSS_WEIGHT_HEATMAP = 10_000

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

X_train, _, y_train, _ = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)


def gaussian_kernel(mu, sigma=0.8, n=48):
    x = np.arange(0, n)
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


y_train_heatmap = np.array([gaussian_kernel(y) for y in y_train])

print(y_train_heatmap.shape)


########## Models ##########
def compile_wrapper(builder, loss_weight):
    def compiled_builder():
        model = builder()
        model.compile(loss='mse', optimizer=optimizers.Adam(LR), loss_weights=loss_weight)
        return model

    return compiled_builder


########## Regular models ##########
print('Regular models...')


def regular_metric(y_true, y_pred):
    std = plot_difference_hist(y_true * TIME_STEP, y_pred[:, 0] * TIME_STEP, show=False, channel=CHANNEL, title=False,
                               print_cov=False, hist_range=(-0.4, 0.4))
    return std * 1000  # ps


regular_model_builders = [
    compile_wrapper(optimal_model_builders.mlp, loss_weight=LOSS_WEIGHT_REGULAR),
    compile_wrapper(optimal_model_builders.convnet, loss_weight=LOSS_WEIGHT_REGULAR)
]
regular_model_names = ['mlp', 'convnet']

cross_validator = CrossValidator(regular_model_builders, X_train, y_train, CROSSVAL_DIR, PROJECT_NAME,
                                 n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, es_min_delta=ES_MIN_DELTA_REGULAR, n_cv=CROSSVAL_N_CV,
                                 n_executions=CROSSVAL_N_EXEC, model_names=regular_model_names,
                                 eval_metric=regular_metric, overwrite=OVERWRITE)

regular_model_scores = cross_validator()

########## Regular models ##########
print('Heatmap models...')


def heatmap_metric(y_heatmap_true, y_heatmap_pred):
    y_true = np.empty(y_heatmap_true.shape[0])
    for i, y in enumerate(y_heatmap_true):
        y_true[i] = _get_gauss_stats(y, std_0=0.8)

    y_pred = np.empty(y_heatmap_pred.shape[0])
    for i, y in enumerate(y_heatmap_pred):
        y_pred[i] = _get_gauss_stats(y, std_0=0.8)

    std = plot_difference_hist(y_true * TIME_STEP, y_pred * TIME_STEP, show=False, channel=CHANNEL, title=False,
                               print_cov=False, hist_range=(-0.4, 0.4))
    return std * 1000  # ps


heatmap_model_builders = [compile_wrapper(optimal_model_builders.unet, loss_weight=LOSS_WEIGHT_HEATMAP)]
heatmap_model_names = ['unet']

cross_validator = CrossValidator(heatmap_model_builders, X_train, y_train_heatmap, CROSSVAL_DIR, PROJECT_NAME,
                                 n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, es_min_delta=ES_MIN_DELTA_HEATMAP, n_cv=CROSSVAL_N_CV,
                                 n_executions=CROSSVAL_N_EXEC, model_names=heatmap_model_names,
                                 eval_metric=heatmap_metric, overwrite=False)

heatmap_model_scores = cross_validator()

########## Comparison ##########
print('Comparison...')

all_model_builders = regular_model_builders + heatmap_model_builders
all_model_names = regular_model_names + heatmap_model_names
all_model_scores = {**regular_model_scores, **heatmap_model_scores}

mean_scores = [f"{np.mean(scores):0.2f}" for scores in all_model_scores.values()]
std_scores = [f"{np.std(scores):0.2f}" for scores in all_model_scores.values()]
n_params = [builder().count_params() for builder in all_model_builders]

df = pd.DataFrame({'mean': mean_scores, 'std': std_scores, 'n_params': n_params}, index=list(all_model_scores.keys()))
df.index.name = 'Model'
print(df)
