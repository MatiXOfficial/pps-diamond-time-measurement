import numpy as np
from matplotlib import pyplot as plt


def plot_sample(ampl: list[float], time: list[float] = None, title: str = '', timestamp: float = None,
                ylim: bool = False, xlabel='time [156.25 ps]', ylabel='voltage [V]', marker=None):
    """
    Plot a waveform
    :param ampl: y axis values
    :param time: x axis values. If None, set to an array from 0 to len(ampl) - 1
    :param title:
    :param timestamp: a vertical line to mark. If None, no line is plotted
    :param ylim: if True, ylim is set to (-0.1, 1.1)
    :param marker: Plot marker
    :return:
    """
    if time is None:
        time = np.arange(len(ampl))
    plt.plot(time, ampl, marker=marker)
    if ylim:
        plt.ylim(-0.1, 1.1)
    if timestamp is not None:
        plt.axvline(timestamp, c='red')
        title += f', t={timestamp:0.2f}'
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_history(history: dict[str, np.array], title: str, ymax: float = None, figsize: tuple[float, float] = (8, 5.5),
                 savefig: bool = None, plt_font_size: int = None):
    """
    Plot the loss history from training a neural network
    :param history: dictionary with the data; history = model.fit(...).history
    :param title: plot title
    :param ymax: maximum of the plot
    :param figsize: figure size
    :param savefig: If else than None, plt will be saved with savefig as the path
    :param plt_font_size: If not None, plt font size will be set
    :return:
    """
    if plt_font_size is not None:
        plt.rc('font', size=plt_font_size)
    plt.figure(figsize=figsize)

    X = np.arange(1, len(history['loss']) + 1)

    plt.plot(X, history['loss'], label='train')
    plt.plot(X, history['val_loss'], label='validation')

    if ymax is not None:
        plt.ylim(0, ymax)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.title(f"val loss {history[f'val_loss'].values[-1]:0.4f} (min: {min(history[f'val_loss'].values):0.4f})")
    plt.title(f"validation loss: {min(history[f'val_loss'].values):0.4f}")
    plt.grid()
    plt.legend()

    plt.suptitle(title)
    
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
    plt.show()
