import numpy as np
from matplotlib import pyplot as plt


def plot_sample(ampl: list[float], time: list[float] = None, title: str = '', timestamp: float = None,
                ylim: bool = False):
    """
    Plot a waveform
    :param ampl: y axis values
    :param time: x axis values. If None, set to an array from 0 to len(ampl) - 1
    :param title:
    :param timestamp: a vertical line to mark. If None, no line is plotted
    :param ylim: if True, ylim is set to (-0.1, 1.1)
    :return:
    """
    if time is None:
        time = np.arange(len(ampl))
    plt.plot(time, ampl)
    if ylim:
        plt.ylim(-0.1, 1.1)
    if timestamp is not None:
        plt.axvline(timestamp, c='red')
        title += f', t={timestamp:0.2f}'
    plt.title(title)


def plot_history(history: dict[str, np.array], title: str, ymax: float = None):
    """
    Plot the loss history from training a neural network
    :param history: dictionary with the data; history = model.fit(...).history
    :param title: plot title
    :param ymax: maximum of the plot
    :return:
    """
    plt.figure(figsize=(10, 7))

    X = np.arange(1, len(history['loss']) + 1)

    plt.plot(X, history['loss'], label='train')
    plt.plot(X, history['val_loss'], label='test')

    if ymax is not None:
        plt.ylim(0, ymax)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f"val loss {history[f'val_loss'].values[-1]:0.4f} (min: {min(history[f'val_loss'].values):0.4f})")
    plt.grid()
    plt.legend()

    plt.suptitle(title)
    plt.show()
