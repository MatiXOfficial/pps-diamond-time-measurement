import sys
from pathlib import Path
from dataclasses import dataclass
import itertools

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.plot_utils import plot_history
from src.cfd_utils import TIME_STEP, plot_diff_hist_stats, _get_gauss_stats

PWD = '.'
CHANNEL = 30
N_BASELINE = 20

########## Load parameters ##########
print('Loading parameters...')
@dataclass
class Parameters:
    UNET_DEPTH: int = 4
    FILTER_MULT: int = 1
    KERNEL_SIZE: int = 3
    ACTIVATION_FUNC: str = 'relu'
    USE_RESIDUAL_BLOCK: bool = False

    def __str__(self) -> str:
        return f"{self.UNET_DEPTH}, {self.FILTER_MULT}, {self.KERNEL_SIZE}, {self.ACTIVATION_FUNC}, {self.USE_RESIDUAL_BLOCK}"
    
unet_depth_list = [3, 4]
filter_mult_list = [1, 2, 4]
kernel_size_list = [3, 5]
activation_func_list = ['relu', 'gelu']  # relu and elu better than elu
use_residual_block_list = [False]  # True much worse than False

parameter_combinations = itertools.product(unet_depth_list, filter_mult_list, kernel_size_list, activation_func_list, use_residual_block_list)

parameter_sets = [Parameters(*params) for params in parameter_combinations]

i_param = int(sys.argv[1])

PARAMETER_SET = parameter_sets[i_param]
print(f'Loaded parameter set: {PARAMETER_SET}')

OUTPUT_PATH = f'/scripts/unet_tuning_ch30/output/{i_param}'
RESULTS_PATH = '/scripts/unet_tuning_ch30/output/'
WEIGHTS_PATH = f'/scripts/model_weights/unet_tuning_ch30/{i_param}'

Path(PWD + OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

########## Load data ##########
print('Loading dataset...')
dataset = np.load(PWD + f'/data/dataset.npz', allow_pickle=True)

all_channels_data = dataset['dataset'].flat[0]
all_channels_data.keys()

all_X, all_y = all_channels_data[CHANNEL][0], all_channels_data[CHANNEL][1]

print(all_X.shape, all_y.shape)

########## Preprocess ##########
print('Preprocessing...')
def augmentation_random_cut(X, y, n_edge_cut=8, seed=None, apply=True):
    if not apply:
        return X, y
    random_state = np.random.RandomState(seed)
    n_to_cut = n_edge_cut * 2
    
    start = random_state.randint(low=0, high=n_to_cut, size=X.shape[0])
    end = X.shape[1] - n_to_cut + start
    
    X_new = np.empty(shape=(X.shape[0], X.shape[1] - n_to_cut))
    for i in range(X.shape[0]):
        X_new[i] = X[i, start[i]:end[i]]
        
    y_new = y - start
    return X_new, y_new

def gaussian_kernel(mu, sigma=0.8, n=48):
    x = np.arange(0, n)
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

all_X -= np.mean(all_X[:, :N_BASELINE], axis=1)[:, None]

all_X /= all_X.max(axis=1)[:, None]

X_aug, y_aug = augmentation_random_cut(all_X, all_y, 8, seed=42, apply=True)

# Don't use the test sets in tuning
X_train, _, y_train, _ = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  # Validation split

Y_heatmap_train = np.array([gaussian_kernel(y) for y in y_train])
Y_heatmap_test = np.array([gaussian_kernel(y) for y in y_test])

print('Preprocessed data')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

########## Model ##########
def residual_block(x, n_filters, strides):
    skip = layers.Conv1D(n_filters, 1, strides=strides)(x)
    
    x = layers.Conv1D(n_filters // 4, 1, strides=strides)(x)
    x = layers.Activation(PARAMETER_SET.ACTIVATION_FUNC)(x)
    
    x = layers.Conv1D(n_filters // 4, PARAMETER_SET.KERNEL_SIZE, padding='same')(x)
    x = layers.Activation(PARAMETER_SET.ACTIVATION_FUNC)(x)
    
    x = layers.Conv1D(n_filters, 1)(x)
    
    x = layers.Add()([skip, x])
    x = layers.Activation(PARAMETER_SET.ACTIVATION_FUNC)(x)
    
    return x

def conv_block(x, n_filters):
    if not PARAMETER_SET.USE_RESIDUAL_BLOCK:
        x = layers.Conv1D(n_filters, PARAMETER_SET.KERNEL_SIZE, activation=PARAMETER_SET.ACTIVATION_FUNC, padding='same')(x)
        skip = layers.Conv1D(n_filters, PARAMETER_SET.KERNEL_SIZE, activation=PARAMETER_SET.ACTIVATION_FUNC, padding='same')(x)
    else:
        skip = residual_block(x, n_filters, strides=1)
    x = layers.MaxPooling1D()(skip)
    return skip, x
    
def deconv_block(x, skip, n_filters):
    x = layers.UpSampling1D()(x)
    x = layers.Conv1D(n_filters, 1, activation='linear')(x)
    x = layers.Concatenate()([skip, x])
    if not PARAMETER_SET.USE_RESIDUAL_BLOCK:
        x = layers.Conv1D(n_filters, PARAMETER_SET.KERNEL_SIZE, activation=PARAMETER_SET.ACTIVATION_FUNC, padding='same')(x)
        x = layers.Conv1D(n_filters, PARAMETER_SET.KERNEL_SIZE, activation=PARAMETER_SET.ACTIVATION_FUNC, padding='same')(x)
    else:
        x = residual_block(x, n_filters, strides=1)
    return x

inputs = layers.Input(48)
x = layers.Reshape((-1, 1))(inputs)

# Example for UNET_DEPTH=4
# skip1, x = conv_block(x, 8 * PARAMETER_SET.FILTER_MULT)
# skip2, x = conv_block(x, 16 * PARAMETER_SET.FILTER_MULT)
# skip3, x = conv_block(x, 32 * PARAMETER_SET.FILTER_MULT)
# x, _ = conv_block(x, 64 * PARAMETER_SET.FILTER_MULT)
# 
# x = deconv_block(x, skip3, 32 * PARAMETER_SET.FILTER_MULT)
# x = deconv_block(x, skip2, 16 * PARAMETER_SET.FILTER_MULT)
# x = deconv_block(x, skip1, 8 * PARAMETER_SET.FILTER_MULT)

n_filters = 8

# encoder
skips = []
for _ in range(PARAMETER_SET.UNET_DEPTH - 1):
    skip, x = conv_block(x, n_filters * PARAMETER_SET.FILTER_MULT)
    skips.append(skip)
    n_filters *= 2

# bottleneck
x, _ = conv_block(x, n_filters * PARAMETER_SET.FILTER_MULT)

# decoder
for _ in range(PARAMETER_SET.UNET_DEPTH - 1):
    n_filters /= 2
    x = deconv_block(x, skips.pop(), n_filters * PARAMETER_SET.FILTER_MULT)

x = layers.Conv1D(1, 1, activation='linear')(x)

outputs = layers.Flatten()(x)

model = tf.keras.Model(inputs, outputs)

########## Train ##########
print("Training...")

def train_model(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, lr=0.001, name='model', train=False, n_epochs=1000, verbose=1, batch_size=2048, lr_patience=None, es_patience=None, loss_weights=None):
    pwd = PWD + WEIGHTS_PATH + f'/{name}/'

    model.compile(loss='mse', optimizer=optimizers.Adam(lr), loss_weights=loss_weights)

    model_callbacks = []
    model_callbacks.append(callbacks.ModelCheckpoint(filepath=pwd + 'weights', save_best_only=True, save_weights_only=True))
    if es_patience is not None:
        model_callbacks.append(callbacks.EarlyStopping(patience=es_patience))
    if lr_patience is not None:
        model_callbacks.append(callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=lr_patience))
    
    if train:
        history = model.fit(X_train, y_train, epochs=n_epochs, verbose=verbose, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=model_callbacks).history
        pd.DataFrame(history).to_csv(pwd + 'loss_log.csv')

    model.load_weights(pwd + 'weights')
    history = pd.read_csv(pwd + 'loss_log.csv')
    
    return history

lr = 0.001
if PARAMETER_SET.ACTIVATION_FUNC == 'gelu':
    lr = 0.002

history = train_model(model, y_train=Y_heatmap_train, y_test=Y_heatmap_test, lr=lr, name='unet_model', train=False, n_epochs=1500, lr_patience=40, es_patience=100, loss_weights=100, verbose=2)

########## Model summary ##########
print("Model summary")
print(model.summary())

########## Predict ##########
print('Predict...')
Y_heatmap_pred = model.predict(X_test)

y_pred = np.empty(Y_heatmap_pred.shape[0])
for i, y in enumerate(Y_heatmap_pred):
    y_pred[i] = _get_gauss_stats(y)

print('MSE:', mean_squared_error(y_test, y_pred))

########## Validate ##########
print("Validate...")

def plot_difference_hist(y_true, y_pred, channel, hist_range=(-2, 2), n_bins=100, xlabel=None, savefig=None, title=True, ymax=None):
    plt.rc('font', size=12)
    mu, std, pcov = plot_diff_hist_stats(y_true, y_pred, show=False, n_bins=n_bins, hist_range=hist_range, hist_label=f'', plot_gauss=True, return_gauss_stats=True, return_pcov=True)

    if title:
        plt.title(f'Diff histogram (channel={channel}), mean={mu:0.3f}, std={std:0.3f}')
    if xlabel is not None:
        plt.xlabel(xlabel)
        
    if ymax is not None:
        plt.ylim(top=ymax)
        
    plt.grid()
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        
    plt.show()
    # print('Covariance matrix of the Gaussian fit:')
    # print(pcov)
        
    return std

plt.close()
plot_history(history, "Test UNet Model", ymax=0.6, savefig=PWD + OUTPUT_PATH + '/test_unet_model_history.png')

plt.close()
std = plot_difference_hist(y_test * TIME_STEP, y_pred * TIME_STEP, CHANNEL, hist_range=(-0.4, 0.4), xlabel='time [ns]', savefig=PWD + OUTPUT_PATH + '/test_unet_model_diff_hist.png')

std_stat = np.std((y_pred - y_test) * TIME_STEP)

print('std:', f'{std*1000:0.2f}')
print('std_stat:', f'{std_stat*1000:0.2f}')

########## Save the result ##########
with open(PWD + RESULTS_PATH + '/results.csv', 'a') as file:
    file.write(f"{i_param}, {PARAMETER_SET}, {std*1000:0.2f}, {std_stat*1000:0.2f}\n")
