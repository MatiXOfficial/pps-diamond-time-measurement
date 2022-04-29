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
