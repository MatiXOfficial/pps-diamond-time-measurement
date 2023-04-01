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
CHANNEL = 17
N_BASELINE = 20

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

X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)

Y_heatmap_train = np.array([gaussian_kernel(y) for y in y_train])
Y_heatmap_test = np.array([gaussian_kernel(y) for y in y_test])

print('Preprocessed data')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

########## Model ##########
def conv_block(x, n_filters):
    x = layers.Conv1D(n_filters, 3, activation='relu', padding='same')(x)
    skip = layers.Conv1D(n_filters, 3, activation='relu', padding='same')(x)
    # skip = residual_block(x, n_filters, strides=1)
    x = layers.MaxPooling1D()(skip)
    return skip, x
    
def deconv_block(x, skip, n_filters):
    x = layers.UpSampling1D()(x)
    x = layers.Conv1D(n_filters, 1, activation='linear')(x)
    x = layers.Concatenate()([skip, x])
    x = layers.Conv1D(n_filters, 3, activation='relu', padding='same')(x)
    x = layers.Conv1D(n_filters, 3, activation='relu', padding='same')(x)
    # x = residual_block(x, n_filters, strides=1)
    return x

inputs = layers.Input(48)
x = layers.Reshape((-1, 1))(inputs)

skip1, x = conv_block(x, 8)
skip2, x = conv_block(x, 16)
skip3, x = conv_block(x, 32)
x, _ = conv_block(x, 64)

x = deconv_block(x, skip3, 32)
x = deconv_block(x, skip2, 16)
x = deconv_block(x, skip1, 8)

x = layers.Conv1D(1, 1, activation='linear')(x)

outputs = layers.Flatten()(x)

model = tf.keras.Model(inputs, outputs)

########## Train ##########
print("Training...")

def train_model(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, lr=0.001, name='model', train=True, n_epochs=1000, verbose=1, batch_size=2048, lr_patience=None, es_patience=None, loss_weights=None):
    pwd = PWD + f'/scripts/model_weights/test/{name}/'

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

history = train_model(model, y_train=Y_heatmap_train, y_test=Y_heatmap_test, lr=0.001, name='test_unet_model', train=True, n_epochs=2000, lr_patience=50, es_patience=200, loss_weights=100, verbose=2)

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
plot_history(history, "Test UNet Model", ymax=0.4, savefig=PWD + '/scripts/test/output/test_unet_model_history.png')

plt.close()
std = plot_difference_hist(y_test * TIME_STEP, y_pred * TIME_STEP, CHANNEL, hist_range=(-0.4, 0.4), xlabel='time [ns]', savefig=PWD + '/scripts/test/output/test_unet_model_diff_hist.png')

print('std:', f'{std*1000:0.2f}')
